"""Validate palmar ETac on the local MuJoCo LEAP Hand."""

from dataclasses import dataclass
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

from . import blind_grasping_assets
from .mujoco_leap import MujocoLeapETac, NODE_SPACING_M, PALMAR_DIRECTIONS, SURFACE_GEOMS


LEAP_XML = Path(__file__).resolve().parents[1] / "src/mjlab/asset_zoo/robots/leaphand_only.xml"
PROBE_HALF_SIZE = 0.006
RETRACT_M = 0.016
INDENT_M = 0.00065
SCAN_DIRECTIONS = (
    (0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 1, 0),
)


def load_model(*, probe=False):
    # Relative paths keep MuJoCo asset loading independent of the Windows code page.
    previous_dir = Path.cwd()
    try:
        os.chdir(LEAP_XML.parent)
        spec = mujoco.MjSpec.from_file(LEAP_XML.name)
        spec.modelname = "LEAP ETac Palmar Deformation" if probe else "LEAP Palmar ETac"
        pads = {body.name for body in spec.bodies if body.name.endswith("_fsr")}
        for exclusion in list(spec.excludes):
            if exclusion.bodyname1 in pads or exclusion.bodyname2 in pads:
                spec.delete(exclusion)
        for name in sorted(pads):
            spec.delete(spec.body(name))
        for mesh in list(spec.meshes):
            if mesh.name.endswith("_fsr_mesh"):
                spec.delete(mesh)

        if probe:
            spec.delete(spec.body("object_body"))
            body = spec.worldbody.add_body(name="probe")
            body.add_freejoint(name="probe_free")
            body.add_geom(
                name="object_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[PROBE_HALF_SIZE] * 3, mass=0.01,
            )
        return spec.compile()
    finally:
        os.chdir(previous_dir)


@dataclass(frozen=True)
class ProbeContact:
    link: int
    point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    quaternion: np.ndarray


class SurfaceProbe:
    """Move a box over the actual mesh while holding the LEAP joint pose."""

    def __init__(self, model, data, reader):
        self.model = model
        self.data = data
        self.reader = reader
        joint = model.joint("probe_free")
        pos, vel = joint.qposadr[0], joint.dofadr[0]
        self._hand_pose = data.qpos[:pos]
        self._rest_pose = self._hand_pose.copy()
        self._hand_velocity = data.qvel[:vel]
        self._probe_pose = data.qpos[pos : pos + 7]
        self._probe_velocity = data.qvel[vel : vel + 6]
        self.park()
        self.points, self.normals = reader.surface_nodes(data)

    def park(self):
        self._hand_pose[:] = self._rest_pose
        self._hand_velocity[:] = 0
        self._probe_pose[:] = [0, 0, 1, 1, 0, 0, 0]
        self._probe_velocity[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def place(self, target, slide=0.0, gap=-INDENT_M, tangent_speed=0.0, normal_speed=0.0):
        self._hand_pose[:] = self._rest_pose
        self._hand_velocity[:] = 0
        self._probe_pose[:3] = (
            target.point + slide * target.tangent + (PROBE_HALF_SIZE + gap) * target.normal
        )
        self._probe_pose[3:] = target.quaternion
        self._probe_velocity[:3] = tangent_speed * target.tangent + normal_speed * target.normal
        self._probe_velocity[3:] = 0

    def contacts(self):
        result = {}
        for contact in self.data.contact:
            geom1, geom2 = contact.geom
            if contact.dist <= 0 and self.reader.object_id in (geom1, geom2):
                other = geom2 if geom1 == self.reader.object_id else geom1
                result[other] = min(result.get(other, 0.0), contact.dist)
        return result

    def find_contact(self, link, directions=SCAN_DIRECTIONS, *, slide=0.0):
        """Find a face with native MuJoCo contact along the requested probe sweep."""
        nodes = self.reader.link_slices[link]
        points, normals = self.points[nodes], self.normals[nodes]
        geom_id = self.reader.geom_ids[link]
        center = self.data.geom_xpos[geom_id].copy()
        surface_scale = np.ptp(points, axis=0)
        for direction in directions:
            normal = np.asarray(direction, dtype=np.float64)
            axes = np.flatnonzero(normal == 0)
            tangent = np.eye(3)[axes[surface_scale[axes].argmax()]]
            rotation = np.column_stack((tangent, np.cross(normal, tangent), normal))
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, rotation.ravel())
            height = (points - center) @ normal
            lateral = points - center - height[:, None] * normal
            rank = height - 0.65 * np.linalg.norm(lateral, axis=-1)
            candidates = np.flatnonzero(normals @ normal > 0.72)
            candidates = candidates[np.argsort(rank[candidates])[::-1]][:110]
            offsets = np.linspace(-slide, slide, 5) if slide else (0,)
            for index in candidates:
                target = ProbeContact(link, points[index], tangent, normal, quaternion.copy())
                for offset in offsets:
                    self.place(target, slide=offset)
                    mujoco.mj_forward(self.model, self.data)
                    distances = self.contacts()
                    if not -0.004 < distances.get(geom_id, 0.0) < -0.00010:
                        break
                else:
                    self.place(target, gap=RETRACT_M)
                    mujoco.mj_forward(self.model, self.data)
                    if not self.contacts():
                        return target
        raise RuntimeError(f"Could not find a probe sweep for {SURFACE_GEOMS[link]}")


def main():
    checkpoints = blind_grasping_assets() / "propagation-ckpt"
    model = load_model()
    reader = MujocoLeapETac(model, checkpoints)
    assert model.nu == 16 and len(reader.geom_ids) == 17
    assert all("_fsr" not in model.geom(i).name for i in range(model.ngeom))
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        active, features = reader.read(data)
        assert active.shape == (sum(reader.counts),)
        assert features.shape == (sum(reader.counts), 3)
        assert torch.isfinite(features).all()

    model = load_model(probe=True)
    data = mujoco.MjData(model)
    reader = MujocoLeapETac(model, checkpoints)
    probe = SurfaceProbe(model, data, reader)
    counts = []
    for link, nodes in enumerate(reader.link_slices):
        directions = (PALMAR_DIRECTIONS[link],) + SCAN_DIRECTIONS
        target = probe.find_contact(link, directions)
        probe.place(target)
        mujoco.mj_forward(model, data)
        mask, features = reader.read(data)
        assert reader.geom_ids[link] in probe.contacts(), SURFACE_GEOMS[link]
        assert mask[nodes].any() and features[nodes].any(), SURFACE_GEOMS[link]
        assert torch.isfinite(features).all()
        indented, offset = reader.deformation(data)
        assert indented[nodes].any() and offset[nodes].any(), SURFACE_GEOMS[link]
        assert torch.isfinite(offset).all()
        pressed = indented.cpu().numpy()
        moved = probe.points[pressed] + offset[pressed].cpu().numpy()
        box_rotation = data.geom_xmat[reader.object_id].reshape(3, 3)
        in_box_frame = (moved - data.geom_xpos[reader.object_id]) @ box_rotation
        face_distance = np.abs(in_box_frame) - model.geom_size[reader.object_id]
        np.testing.assert_allclose(face_distance.max(axis=1), 0, atol=1e-6)
        counts.append(int(mask[nodes].sum()))

    # A probe pressing the opposite face still collides with the hand in MuJoCo.
    palm = reader.geom_ids[0]
    normal = np.array([0., 0., 1.])
    start = data.geom_xpos[palm] + 0.10 * normal
    distance = mujoco.mj_rayMesh(model, data, palm, start, -normal)
    assert distance >= 0
    back = ProbeContact(0, start - distance * normal, np.array([1., 0., 0.]), normal,
                        np.array([1., 0., 0., 0.]))
    probe.place(back)
    mujoco.mj_forward(model, data)
    assert palm in probe.contacts()
    mask, features = reader.read(data)
    assert not mask.any() and not features.any()
    indented, offset = reader.deformation(data)
    assert not indented.any() and not offset.any()

    probe.park()
    mask, features = reader.read(data)
    assert not mask.any() and not features.any()
    indented, offset = reader.deformation(data)
    assert not indented.any() and not offset.any()
    print(f"LEAP ETac on {torch.cuda.get_device_name()}")
    print(f"{sum(reader.counts):,} palmar nodes at {1000 * NODE_SPACING_M:g} mm pitch across 17 links")
    print(f"All 17 links responded to physical probe contact ({min(counts)}-{max(counts)} active nodes)")
    print("Indented nodes moved to the box surface in 3D.")
    print("The uninstrumented dorsal surface collided without a tactile signal. Separation returned zero.")


if __name__ == "__main__":
    main()
