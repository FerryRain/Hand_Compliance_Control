"""Check the Windows ETac runtime against the released ShadowHand and cube."""

import json
import os
from pathlib import Path

import mujoco
import torch
from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

from . import blind_grasping_assets
from .etac import ETac


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This ETac build requires CUDA.")

    device = torch.device("cuda:0")
    assets = blind_grasping_assets()
    # The bundled MJCF parser resolves mesh paths from the working directory.
    launch_directory = Path.cwd()
    try:
        os.chdir(assets)
        simulator = ETac(
            hand_file_path=assets / "mjcf" / "shadow_hand_vis.xml",
            hand_mesh_path=assets / "mjcf" / "meshes",
            object_root_path=assets / "data" / "meshdata",
            num_points=0,
            batch_size=2,
            device=device,
        )
    finally:
        os.chdir(launch_directory)
    checkpoints = assets / "propagation-ckpt"
    simulator.setup_propagation(
        [checkpoints / "linear.pth", checkpoints / "resdualnet.pth"],
        [checkpoints / "linear-ft.pth", checkpoints / "resdualnet-ft.pth"],
    )
    simulator.set_object("cube")

    layout = json.loads((assets / "points-1000-0903.txt").read_text(encoding="utf-8"))
    hand = simulator.hand_model
    for link, points in layout.items():
        hand.mesh[link]["surface_points"] = torch.tensor(points, dtype=torch.float32, device=device)
    simulator.set_dicts()

    cube = simulator.object_model.object_mesh_list[0]
    vertices = torch.tensor(cube.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(cube.faces, dtype=torch.int64, device=device)
    mesh = Meshes(vertices[None], faces[None])
    candidates = sample_points_from_meshes(mesh, num_samples=16)
    sampled, _ = sample_farthest_points(candidates, K=4)
    assert sampled.shape == (1, 4, 3) and torch.isfinite(sampled).all()

    hand_state = torch.zeros((2, 13), device=device)
    hand_state[:, 6] = 1  # ETac uses xyzw quaternions.
    joint_positions = torch.zeros((2, hand.n_dofs), device=device)
    simulator.set_hand_model_pos(hand_state[:, :7], joint_positions)

    palm = "robot0:palm"
    palm_points = hand.mesh[palm]["surface_points"]
    palm_point = hand.current_status[palm].transform_points(palm_points)[0, 0]
    object_state = torch.zeros_like(hand_state)
    object_state[:, 6] = 1
    face_offset = palm_point.new_tensor([cube.bounds[1, 0] + 0.0005, 0, 0])
    object_state[0, :3] = palm_point - face_offset
    object_state[1, :3] = 5

    link_states = torch.zeros((2, simulator.num_links, 13), device=device)
    link_states[:, :, 6] = 1
    for index, link in enumerate(simulator.ind2link):
        link_states[:, index, :3] = hand.current_status[link].get_matrix()[:, 3, :3]

    counts, contacts, displacement = simulator.cal_tactile_info(
        hand_state, object_state, joint_positions, link_states
    )
    active, separated = counts.tolist()
    assert displacement.shape == (2, 935, 3) and torch.isfinite(displacement).all()
    peak = torch.linalg.vector_norm(displacement[0], dim=-1).max().item()
    assert active > 0 and contacts[0].any() and peak > 0
    assert separated == 0 and torch.count_nonzero(contacts[1]) == 0
    assert torch.count_nonzero(displacement[1]) == 0

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.01"/>'
        '</body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0

    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(device)}")
    print(f"ETac checked {displacement.shape[1]} tactile nodes")
    print(f"Near cube gave {active} active nodes and nonzero tactile output")
    print(f"Separated cube gave {separated} active nodes and zero tactile output")
    print(f"MuJoCo {mujoco.__version__} completed a physics step")


if __name__ == "__main__":
    main()
