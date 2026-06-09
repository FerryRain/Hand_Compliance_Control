import sys
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, list_tasks
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer

# 零动作策略
class ZeroActionPolicy:
    def __init__(self, action_dim: int, device: str):
        self.action_dim = action_dim
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        num_envs = obs.shape[0] if torch.is_tensor(obs) else 1
        return torch.zeros((num_envs, self.action_dim), device=self.device)


def main():
    all_tasks = list_tasks()
    task_id = "Leaphand-Finger-Adhesion-Control"

    if len(sys.argv) > 1 and sys.argv[1] in all_tasks:
        task_id = sys.argv[1]
    else:
        print(f"[INFO] 未指定任务。默认加载: {task_id}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    action_dim = env.action_manager.total_action_dim

    # 扫描编译后模型中的主动吸附执行器
    adhesion_indices = []
    print("\n[INFO] 正在扫描主动吸附执行器...")
    m = env.sim.mj_model  # 原始 MjModel，不是 mjwarp

    for i in range(m.nu):
        try:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        except Exception:
            name = ""
        if name and "adhere" in name:
            adhesion_indices.append(i)
            print(f"  - {name} (index {i})")

    if adhesion_indices:
        print(f"[SUCCESS] 共 {len(adhesion_indices)} 个吸附器，ctrl 已设为 1.0")
    else:
        print("[WARNING] 未找到吸附器！")

    # step 前注入 ctrl=1.0
    original_step = env.step

    def step_with_adhesion(action: torch.Tensor):
        if adhesion_indices:
            if env.sim.data.ctrl.ndim == 2:
                env.sim.data.ctrl[:, adhesion_indices] = 1.0
            else:
                env.sim.data.ctrl[adhesion_indices] = 1.0
        return original_step(action)

    env.step = step_with_adhesion

    viewer_env = RslRlVecEnvWrapper(env)
    policy = ZeroActionPolicy(action_dim=action_dim, device=device)

    print(f"\n[SUCCESS] 已加载: {task_id}")
    print("[TIP] 右键拖蓝色胶囊靠近手指，FSR 吸附器会把手指吸向胶囊表面\n")

    NativeMujocoViewer(viewer_env, policy).run()
    viewer_env.close()


if __name__ == "__main__":
    main()