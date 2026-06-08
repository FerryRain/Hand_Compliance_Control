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
        obj_poses = np.array(f_in["obj_pose"])
        palm_poses = np.array(f_in["palm_pose"])
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
        
        # 准备输出容器
        # 在新坐标系下，obj_pose 永远是 [0,0,0, 1,0,0,0] (原点 + 单位旋转)
        # 我们主要计算新的 palm_pose_inverted
        palm_poses_inv = np.zeros_like(palm_poses)
        
        print(f"[INFO] Processing {num_envs} environments over {num_steps} steps...")

        for e in range(num_envs):
            for t in range(num_steps):
                # 1. 提取当前时刻物体的位姿 T_world_obj
                obj_p = obj_poses[t, e, :3]
                obj_q = obj_poses[t, e, 3:]
                T_w_obj = pose_to_matrix(obj_p, obj_q)
                
                # 2. 提取当前时刻手掌的位姿 T_world_palm
                palm_p = palm_poses[t, e, :3]
                palm_q = palm_poses[t, e, 3:]
                T_w_palm = pose_to_matrix(palm_p, palm_q)
                
                # 3. 核心转换逻辑：计算手掌相对于物体的位姿
                # T_obj_palm = inv(T_w_obj) * T_w_palm
                # 这样当物体被重置到原点时，手掌就在这个相对位姿上
                T_rel = np.linalg.inv(T_w_obj) @ T_w_palm
                
                # 4. 转回 pos, quat 存入结果
                new_p, new_q = matrix_to_pose(T_rel)
                palm_poses_inv[t, e, :3] = new_p
                palm_poses_inv[t, e, 3:] = new_q

        # 保存新文件
        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset("time", data=time_data)
            f_out.create_dataset("fsr", data=fsr_data)
            f_out.create_dataset("q", data=q_data) # 关节角保持不变
            f_out.create_dataset("palm_pose_world", data=palm_poses_inv)
            
            # 物体位姿固定在原点
            fixed_obj = np.zeros_like(obj_poses)
            fixed_obj[:, :, 3] = 1.0 # w=1, x,y,z=0
            f_out.create_dataset("obj_pose_world", data=fixed_obj)
            
            # 也可以把 action 存下来（如果有记录的话）
            if "action" in f_in:
                f_out.create_dataset("action", data=np.array(f_in["action"]))

        print(f"[SUCCESS] Inverted data saved to {output_path}")

if __name__ == "__main__":
    # 使用示例
    import glob
    # 找到最新的 h5 文件
    list_of_files = glob.glob('./finger_copliance_control/data/*.h5')
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        run_inversion(latest_file)
    else:
        print("No H5 files found.")