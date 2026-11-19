"""Run ETac over the palmar surfaces of the MuJoCo LEAP Hand."""

from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

from ETac.utils.models import LinearModel
from .etac import contact_force
from .object_model import ObjectModel


SURFACE_GEOMS = (
    "palm_lower_collision",
    "mcp_joint_geom", "pip_geom", "dip_geom", "fingertip_geom",
    "mcp_joint_2_geom", "pip_2_geom", "dip_2_geom", "fingertip_2_geom",
    "mcp_joint_3_geom", "pip_3_geom", "dip_3_geom", "fingertip_3_geom",
    "pip_4_geom", "thumb_pip_geom", "thumb_dip_geom", "thumb_fingertip_geom",
)
CONTACT_BAND_M = 0.002
NODE_SPACING_M = 0.0008
# The two proximal thumb links face with the palm; the distal links face the fingers.
PALMAR_DIRECTIONS = ((0, 0, -1),) * 15 + ((0, -1, 0),) * 2


def _sample_palmar_mesh(model, rest, geom_id, outward):
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError(f"Expected a mesh for LEAP surface {model.geom(geom_id).name}")
    mesh_id = model.geom_dataid[geom_id]
    first_vertex = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[first_vertex : first_vertex + model.mesh_vertnum[mesh_id]]
    origin, rotation = rest.geom_xpos[geom_id], rest.geom_xmat[geom_id].reshape(3, 3)
    world_vertices = vertices @ rotation.T + origin
    outward = np.asarray(outward, dtype=np.float64)
    tangents = np.flatnonzero(outward == 0)
    grids = [np.arange(world_vertices[:, axis].min() + NODE_SPACING_M / 2,
                       world_vertices[:, axis].max(), NODE_SPACING_M) for axis in tangents]
    ray_origin = ((world_vertices @ outward).max() + 0.003) * outward
    ray_direction = -outward
    normal = np.empty(3)
    points, normals = [], []
    # The nearest hit from the grasp side excludes the back and covered CAD layers.
    for u in grids[0]:
        for v in grids[1]:
            ray_origin[tangents] = (u, v)
            distance = mujoco.mj_rayMesh(model, rest, geom_id, ray_origin, ray_direction, normal)
            if distance >= 0:
                hit = ray_origin + distance * ray_direction
                points.append(hit)
                normals.append(normal.copy() if normal @ outward >= 0 else -normal)
    if not points:
        raise ValueError(f"Could not sample LEAP palmar mesh {model.geom(geom_id).name}")
    return ((np.asarray(points) - origin) @ rotation).astype(np.float32), (np.asarray(normals) @ rotation).astype(np.float32)


class MujocoLeapETac:
    """Evaluate a box against LEAP's palm-facing side across all 17 links."""

    def __init__(self, model: mujoco.MjModel, checkpoint_dir: Path, *, object_geom="object_geom"):
        if not torch.cuda.is_available():
            raise RuntimeError("The ETac signed-distance kernel requires CUDA.")
        self._device = torch.device("cuda")
        self._model = model
        self.geom_ids = np.array([model.geom(name).id for name in SURFACE_GEOMS])
        self.object_id = model.geom(object_geom).id
        if model.geom_type[self.object_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"Expected MuJoCo box geometry for {object_geom}")

        rest = mujoco.MjData(model)
        mujoco.mj_forward(model, rest)
        samples = [_sample_palmar_mesh(model, rest, gid, outward)
                   for gid, outward in zip(self.geom_ids, PALMAR_DIRECTIONS)]
        points, normals = zip(*samples)
        self.counts = np.array([len(p) for p in points])
        edges = np.r_[0, self.counts.cumsum()]
        self.link_slices = tuple(slice(start, stop) for start, stop in zip(edges[:-1], edges[1:]))
        self._local_points = np.concatenate(points)
        self._local_normals = np.concatenate(normals)
        self._node_links = np.repeat(np.arange(len(self.geom_ids)), self.counts)
        self._gpu_points = self._tensor(self._local_points)
        self._gpu_links = torch.as_tensor(self._node_links, device=self._device)

        box = trimesh.creation.box(extents=2 * model.geom_size[self.object_id])
        self._object_model = ObjectModel(None, batch_size_each=1, device=self._device)
        self._object_model.initialize_mesh(box)
        self._propagation = LinearModel().to(self._device)
        weights = torch.load(Path(checkpoint_dir) / "linear.pth", map_location="cpu", weights_only=True)
        self._propagation.load_state_dict(weights)

    def _tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self._device)

    def _hand_poses(self, data):
        if data.model is not self._model:
            raise ValueError("MjData must belong to the model used to create the LEAP adapter.")
        return data.geom_xpos[self.geom_ids], data.geom_xmat[self.geom_ids].reshape(-1, 3, 3)

    def surface_nodes(self, data: mujoco.MjData):
        """Return world positions and surface normals, ordered by link_slices."""
        centers, rotations = self._hand_poses(data)
        rotations = rotations[self._node_links]
        points = np.einsum("nij,nj->ni", rotations, self._local_points)
        normals = np.einsum("nij,nj->ni", rotations, self._local_normals)
        return points + centers[self._node_links], normals

    def _query(self, data):
        centers, rotations = self._hand_poses(data)
        node_centers = self._tensor(centers)[self._gpu_links]
        node_rotations = self._tensor(rotations)[self._gpu_links]
        hand_lever = (node_rotations @ self._gpu_points.unsqueeze(-1)).squeeze(-1)
        world_points = node_centers + hand_lever

        object_center = self._tensor(data.geom_xpos[self.object_id])
        object_rotation = self._tensor(data.geom_xmat[self.object_id].reshape(3, 3))
        object_points = (world_points - object_center) @ object_rotation
        distance, closest, normals = self._object_model.cal_distance(object_points.unsqueeze(0))
        return node_rotations, hand_lever, object_rotation, object_points, distance[0], closest[0], normals[0]

    def _propagate(self, active, local_vectors):
        result = torch.zeros_like(self._gpu_points)
        active_links = self._gpu_links[active]
        active_points = self._gpu_points[active]
        for link in active_links.unique().tolist():
            selected = active_links == link
            sources = active_points[selected]
            vectors = local_vectors[selected]
            receivers = self._gpu_points[self.link_slices[link]]
            values = []
            for chunk in receivers.split(1024):
                distances = torch.cdist(chunk, sources, compute_mode="donot_use_mm_for_euclid_dist")
                values.append(self._propagation.kernel(distances) @ vectors)
            result[self.link_slices[link]] = torch.cat(values)
        return result

    @torch.no_grad()
    def read(self, data: mujoco.MjData):
        """Return the active-node mask [N] and link-local tactile features [N, 3]."""
        node_rotations, hand_lever, object_rotation, _, distance, closest, normals = self._query(data)
        active = distance.abs() < CONTACT_BAND_M
        if not active.any():
            return active, torch.zeros_like(self._gpu_points)

        # MuJoCo returns angular and linear velocity in world axes about each geom center.
        geom_ids = [*self.geom_ids, self.object_id]
        spatial = np.empty((len(geom_ids), 6))
        for geom_id, velocity in zip(geom_ids, spatial):
            mujoco.mj_objectVelocity(self._model, data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, velocity, 0)
        spatial = self._tensor(spatial)
        active_links = self._gpu_links[active]
        surface_motion = spatial[active_links]
        hand_velocity = surface_motion[:, 3:] + torch.linalg.cross(surface_motion[:, :3], hand_lever[active])
        object_lever = closest[active] @ object_rotation.T
        object_motion = spatial[-1].expand(len(object_lever), -1)
        object_velocity = object_motion[:, 3:] + torch.linalg.cross(object_motion[:, :3], object_lever)
        world_normals = normals[active] @ object_rotation.T
        depths = distance[active] + CONTACT_BAND_M
        world_forces = contact_force(depths, world_normals, hand_velocity - object_velocity)
        local_forces = (node_rotations[active].transpose(-1, -2) @ world_forces.unsqueeze(-1)).squeeze(-1)

        return active, self._propagate(active, local_forces)

    @torch.no_grad()
    def deformation(self, data: mujoco.MjData):
        """Return indented nodes [N] and their propagated world displacements [N, 3] in meters."""
        rotations, _, object_rotation, object_points, distance, closest, _ = self._query(data)
        indented = (distance > 0) & (distance < CONTACT_BAND_M)
        if not indented.any():
            return indented, torch.zeros_like(self._gpu_points)

        # The indenter imposes displacement at intersecting nodes. Propagate it along each link.
        contact_shift = (closest[indented] - object_points[indented]) @ object_rotation.T
        local_shift = (rotations[indented].transpose(-1, -2) @ contact_shift.unsqueeze(-1)).squeeze(-1)
        displacement = self._propagate(indented, local_shift)
        displacement[indented] = local_shift
        return indented, (rotations @ displacement.unsqueeze(-1)).squeeze(-1)
