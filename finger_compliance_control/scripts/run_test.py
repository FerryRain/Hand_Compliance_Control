import sys
from dataclasses import asdict
import torch
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, list_tasks
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer


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

    # ── 从注册任务中加载物理控制器 ──
    rl_cfg = load_rl_cfg(task_id)
    policy_class = getattr(rl_cfg, "policy_class", None)
    if policy_class is None:
        raise ValueError(f"任务 '{task_id}' 在其注册的 rl_cfg 中未配置 'policy_class'。")

    cfg_dict = asdict(rl_cfg)
    cfg_dict.pop("policy_class", None)
    cfg_dict.pop("device", None)

    print(f"[INFO] 成功载入注册物理控制器: {policy_class.__name__}")
    print(f"[INFO] enable_fsr_compliance = {cfg_dict.get('enable_fsr_compliance', False)} (默认关闭)")
    policy = policy_class(device=device, num_envs=env.num_envs, **cfg_dict)

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

    # adapt_action_dim: 控制器返回 16-D，需适配 action_manager 实际维度
    def make_policy_with_adapter(raw_policy, target_dim):
        class AdaptedPolicy:
            _step = 0
            def __call__(self, obs):
                action = raw_policy(obs)
                # ── env 0 FSR 输出(每 50 步) ──
                if self._step % 50 == 0:
                    policy_obs = obs["policy"]
                    _f = policy_obs[0, :16].detach().cpu().numpy()
                    print(f"[FSR step {self._step:05d}] palm:{_f[0]:5.1f} {_f[1]:5.1f} {_f[2]:5.1f} {_f[3]:5.1f} | "
                          f"idx:{_f[4]:5.1f} {_f[5]:5.1f} {_f[6]:5.1f} | "
                          f"mid:{_f[7]:5.1f} {_f[8]:5.1f} {_f[9]:5.1f} | "
                          f"ring:{_f[10]:5.1f} {_f[11]:5.1f} {_f[12]:5.1f} | "
                          f"thumb:{_f[13]:5.1f} {_f[14]:5.1f} {_f[15]:5.1f}",
                          flush=True)
                self._step += 1
                current_dim = int(action.shape[-1])
                if current_dim == target_dim:
                    return action
                if current_dim > target_dim:
                    return action[:, :target_dim]
                pad = action.new_zeros((action.shape[0], target_dim - current_dim))
                return torch.cat((action, pad), dim=-1)
        return AdaptedPolicy()

    viewer_env = RslRlVecEnvWrapper(env)
    adapted_policy = make_policy_with_adapter(policy, action_dim)

    print(f"\n[SUCCESS] 已加载: {task_id}")
    print(f"[INFO] 控制器输出维度: 16, 动作空间维度: {action_dim}")
    print("[TIP] 右键拖蓝色胶囊靠近手指，FSR 吸附器会把手指吸向胶囊表面\n")

    NativeMujocoViewer(viewer_env, adapted_policy).run()
    viewer_env.close()


if __name__ == "__main__":
    main()