from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.leaphand.leaphand_mcc_finger_env_cfg import (
    MCCLeapHandPositionControlCfg,
    mcc_finger_contact_env_cfg,
)
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


TASK_ID = "Leaphand-Finger-MCC-Position-Control"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MCC fingertip position control.")
    parser.add_argument("--viewer", choices=("native", "viser"), default="native")
    parser.add_argument("--device", default=None)
    parser.add_argument("--print-every", type=int, default=100)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = mcc_finger_contact_env_cfg(num_envs=1, play=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)

    rl_cfg = MCCLeapHandPositionControlCfg()
    kwargs = asdict(rl_cfg)
    policy_class = kwargs.pop("policy_class")
    kwargs.pop("device", None)
    controller = policy_class(device=device, num_envs=1, **kwargs)

    class DiagnosticPolicy:
        def __init__(self):
            self.step = 0

        def __call__(self, obs):
            action = controller(obs)
            if self.step % max(args.print_every, 1) == 0:
                forces = obs["finger"][0, :12].reshape(4, 3)
                magnitude = torch.linalg.vector_norm(forces, dim=-1)
                debug = controller.last_debug
                ik_error = torch.linalg.vector_norm(
                    debug["tip_x_ik"] - debug["tip_x_ref"], dim=-1
                )[0]
                print(
                    f"[MCC-TIP] step={self.step:05d} "
                    f"force_N={magnitude.detach().cpu().numpy().round(3).tolist()} "
                    f"ik_err_mm={(ik_error.detach().cpu().numpy()*1000).round(2).tolist()} "
                    f"palm_err_mm={float(debug['palm_tracking_error'][0, 0])*1000:.2f}",
                    flush=True,
                )
            self.step += 1
            return action

    policy = DiagnosticPolicy()
    print(f"[INFO] task={TASK_ID} device={device} viewer={args.viewer}")
    print("[INFO] 3-D fingertip force is measured but NOT fed into the controller")
    print("[INFO] Native: use Ctrl+C for the most conservative shutdown path")
    try:
        if args.viewer == "native":
            NativeMujocoViewer(wrapped, policy).run()
        else:
            ViserPlayViewer(wrapped, policy).run()
    finally:
        wrapped.close()
        print("[INFO] Simulation resources released")


if __name__ == "__main__":
    main()
