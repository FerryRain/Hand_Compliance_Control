import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import os

def mujoco_quat_to_scipy(q):
    """将 MuJoCo 的 (w, x, y, z) 转换为 Scipy 的 (x, y, z, w)"""
    return np.array([q[1], q[2], q[3], q[0]])

def scipy_quat_to_mujoco(q):
    """将 Scipy 的 (x, y, z, w) 转换为 MuJoCo 的 (w, x, y, z)"""
    return np.array([q[3], q[0], q[1], q[2]])

def pose_to_matrix(pos, quat_wj):
    """将位置和 MuJoCo 四元数转换为 4x4 变换矩阵"""
    mat = np.eye(4)
    # Scipy Rotation 使用 (x,y,z,w)
    r = R.from_quat(mujoco_quat_to_scipy(quat_wj))
    mat[:3, :3] = r.as_matrix()
    mat[:3, 3] = pos
    return mat

def matrix_to_pose(mat):
    """将 4x4 矩阵转回位置和 MuJoCo 四元数"""
    pos = mat[:3, 3]
    r = R.from_matrix(mat[:3, :3])
    quat_scipy = r.as_quat()
    quat_mj = scipy_quat_to_mujoco(quat_scipy)
    return pos, quat_mj

def run_inversion(input_path, output_path=None):
    if output_path is None:
        output_path = input_path.replace(".h5", "_inverted.h5")

    print(f"[INFO] Reading from {input_path}...")

    with h5py.File(input_path, "r") as f_in:
        # 使用 np.array() 代替 [:]
        obj_poses = np.array(f_in["obj_pose"], dtype=np.float64)
        palm_poses = np.array(f_in["palm_pose"], dtype=np.float64)
        fsr_data = np.array(f_in["fsr"])
        q_data = np.array(f_in["q"])
        time_data = np.array(f_in["time"])

        pos_delta = np.linalg.norm(
            palm_poses[..., :3] - obj_poses[..., :3],
            axis=-1,
        )
        if float(pos_delta.mean()) < 1e-4:
            raise ValueError(
                "Degenerate input: palm_pose is nearly identical to obj_pose. "
                "This usually means collect_data logged the wrong body name. "
                "Re-collect data after fixing palm body selection."
            )

        num_steps, num_envs, _ = obj_poses.shape
        total = num_steps * num_envs
        print(f"[INFO] Processing {num_envs} environments x {num_steps} steps "
              f"= {total:,} frames (vectorized)...")

        # --- 向量化计算 ---
        # obj/palm 位姿: (T, E, 7) -> (px,py,pz, qw,qx,qy,qz)
        obj_p = obj_poses[..., :3]          # (T, E, 3)
        obj_q_mj = obj_poses[..., 3:]       # (T, E, 4)  w,x,y,z
        palm_p = palm_poses[..., :3]
        palm_q_mj = palm_poses[..., 3:]

        # MuJoCo (w,x,y,z) -> scipy (x,y,z,w)
        obj_q_scipy = obj_q_mj[..., [1, 2, 3, 0]].reshape(-1, 4)
        palm_q_scipy = palm_q_mj[..., [1, 2, 3, 0]].reshape(-1, 4)

        obj_R = R.from_quat(obj_q_scipy)        # (T*E,)
        palm_R = R.from_quat(palm_q_scipy)

        obj_R_mat = obj_R.as_matrix()            # (T*E, 3, 3)
        palm_R_mat = palm_R.as_matrix()

        # R_rel = R_obj^T @ R_palm
        R_rel_mat = obj_R_mat.transpose(0, 2, 1) @ palm_R_mat  # (T*E, 3, 3)

        # p_rel = R_obj^T @ (p_palm - p_obj)
        delta_p = (palm_p - obj_p).reshape(-1, 3, 1)            # (T*E, 3, 1)
        p_rel = (obj_R_mat.transpose(0, 2, 1) @ delta_p).squeeze(-1)  # (T*E, 3)
        p_rel = p_rel.reshape(num_steps, num_envs, 3)

        # R_rel -> MuJoCo quat (w,x,y,z)
        R_rel_obj = R.from_matrix(R_rel_mat)
        q_rel_scipy = R_rel_obj.as_quat()                        # (T*E, 4) x,y,z,w
        q_rel_mj = q_rel_scipy[:, [3, 0, 1, 2]].reshape(num_steps, num_envs, 4)

        palm_poses_inv = np.concatenate([p_rel, q_rel_mj], axis=-1).astype(np.float32)

        # 保存新文件
        print(f"[INFO] Writing inverted data to {output_path}...")
        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset("time", data=time_data)
            f_out.create_dataset("fsr", data=fsr_data)
            f_out.create_dataset("q", data=q_data) # 关节角保持不变
            f_out.create_dataset("palm_pose_world", data=palm_poses_inv)

            # 物体位姿固定在原点
            fixed_obj = np.zeros_like(obj_poses, dtype=np.float32)
            fixed_obj[:, :, 3] = 1.0 # w=1, x,y,z=0
            f_out.create_dataset("obj_pose_world", data=fixed_obj)

            # 透传额外字段
            for field in ["action", "fingertip_force_3d", "fingertip_pos", "episode_id",
                          "finger_contact", "contact_stability", "fsr_delta_norm",
                          "finger_force", "full_contact", "force_balance"]:
                if field in f_in:
                    f_out.create_dataset(field, data=np.array(f_in[field]))

        print(f"[SUCCESS] Inverted data saved to {output_path}")

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description="Invert H5 data: transform palm_pose to object-centric frame."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to input .h5 file. If omitted, use the latest in data/headless/.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path. Default: <input>_inverted.h5.",
    )
    args = parser.parse_args()

    if args.file:
        input_path = args.file
    else:
        list_of_files = glob.glob('./finger_compliance_control/data/headless/*.h5')
        # Also search old location
        list_of_files += glob.glob('./finger_compliance_control/data/*.h5')
        # Exclude already-inverted files
        list_of_files = [f for f in list_of_files if "_inverted" not in f]
        if list_of_files:
            input_path = max(list_of_files, key=os.path.getctime)
        else:
            print("No H5 files found.")
            exit(1)

    run_inversion(input_path, args.output)