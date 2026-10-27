"""Deploy one selected episode from a regional request in the native viewer."""
import argparse
import json
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, default=root/'mcc_finger_compliance_control/data/DPPO/regional_force_causality_20261025/batch_05_baseline/source.request.json')
    parser.add_argument('--lane', type=int, default=0,
                        help='Select a row from the source request; only that environment is created')
    parser.add_argument('--effort-limit', type=float, choices=(.4, .8), default=.8)
    parser.add_argument('--render-fps', type=float, default=60.)
    parser.add_argument('--frames', type=int, help='Optional short smoke test; default is the original full length')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    task = json.loads(args.request.read_text())
    if task.get('dppo') or task.get('residual_mode', 'off') == 'train':
        parser.error('Only evaluation requests are permitted')
    if not 0 <= args.lane < len(task['rows']) or args.render_fps <= 0:
        parser.error('Invalid lane or render FPS')
    source_lane = args.lane
    task['rows'] = [task['rows'][source_lane]]
    task['viewer_source_lane'] = source_lane
    args.lane = 0
    if args.frames is not None:
        if args.frames <= 76: parser.error('frames must exceed 76')
        for row in task['rows']: row['frames'] = min(row['frames'], args.frames)
    output_root = Path(task['output']).resolve().parent/'viewer'
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='view_', dir=output_root))
    task.update(output=str(output/'rollout.json'), persist_trace=False)
    request = output/'request.json'
    request.write_text(json.dumps(task, indent=2)+'\n')

    import torch
    import deploy_b_gpu_serial_parity as deployment
    import evaluate_regional_dp as evaluation
    from mjlab.envs import ManagerBasedRlEnv
    from batched_b_control_chain import BatchedBControlChain
    from collection_force_regulation import CollectionForceObserver
    from visualize_dp0_gpu import ContactDisplay, DrivenNativeViewer, RestartReplay

    viewer = None
    class Display(ContactDisplay):
        def refresh(self):
            lane = args.lane if viewer is None else viewer.env_idx
            if hasattr(self, 'full'):
                for key, value in self.full.items(): setattr(self, key, value[lane:lane+1])
                self.active = bool(self.active_lanes[lane])
            if hasattr(self, 'all_contacts'):
                self.contacts = {k:v[lane:lane+1] for k,v in self.all_contacts.items()}
        def draw(self, visualizer):
            self.refresh(); super().draw(visualizer)
        def labels(self, frames):
            self.refresh(); return super().labels(frames)
    display = Display()
    old_step, old_close = ManagerBasedRlEnv.step, ManagerBasedRlEnv.close
    old_chain, old_read = BatchedBControlChain.step, CollectionForceObserver.read
    old_cfg = deployment.replay_env_cfg
    old_pad = evaluation.pad_episode
    if args.frames is not None:
        evaluation.pad_episode = lambda values, length: old_pad(values[:length], length)
    def cfg(**kw):
        kw['hand_effort_limit'] = args.effort_limit
        return old_cfg(**kw)
    def read(observer, *a, **kw):
        result = old_read(observer, *a, **kw)
        display.all_contacts = observer.latest
        return result
    def chain_step(chain, **kw):
        action, diag = old_chain(chain, **kw)
        display.capture(chain, kw, diag)
        display.full = {k:getattr(display,k) for k in ('q','pose','nominal','normal')}
        display.active_lanes = chain.context['active'].clone()
        return action, diag
    def step(env, action):
        nonlocal viewer
        if not args.headless and viewer is None:
            c = env.cfg.viewer
            c.env_idx=args.lane; c.distance=.45; c.azimuth=45.; c.elevation=-10.
            c.origin_type=c.OriginType.ASSET_BODY; c.entity_name='robot'; c.body_name='palm_lower'
            original_visualizers = env.update_visualizers
            def draw(v): original_visualizers(v); display.draw(v)
            env.update_visualizers = draw
            viewer = DrivenNativeViewer(env, display, max(r['frames'] for r in task['rows']), args.render_fps)
            viewer.setup()
        if viewer is not None: viewer.before_step()
        result = old_step(env, action)
        if hasattr(display, 'full'):
            robot = env.scene['robot']
            display.full['q'] = robot.data.joint_pos.clone()
            display.full['pose'] = robot.data.root_link_pose_w.clone()
        if viewer is not None: viewer.after_step()
        return result
    def close(env):
        nonlocal viewer
        if viewer is not None: viewer.close(); viewer=None
        return old_close(env)
    deployment.replay_env_cfg=cfg
    ManagerBasedRlEnv.step=step; ManagerBasedRlEnv.close=close
    BatchedBControlChain.step=chain_step; CollectionForceObserver.read=read
    print(f'[VIEW] lane={args.lane} episode={task["rows"][args.lane]["episode_id"]}; '
          f'single environment (source lane {source_lane}); torque={args.effort_limit} Nm; CUDA; record-and-continue above100N; output={output}', flush=True)
    print('[VIEW] Space pause; Right single step; +/- speed; Enter restart; R markers.', flush=True)
    try:
        while True:
            try:
                evaluation._worker(request)
                break
            except RestartReplay:
                deployment.replay_env_cfg=cfg
                display=Display()
            import gc
            gc.collect(); torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print('[VIEW] Closed/interrupted.', flush=True)
    finally:
        ManagerBasedRlEnv.step=old_step; ManagerBasedRlEnv.close=old_close
        BatchedBControlChain.step=old_chain; CollectionForceObserver.read=old_read
        deployment.replay_env_cfg=old_cfg
        evaluation.pad_episode=old_pad
        if viewer is not None: viewer.close()


if __name__ == '__main__': main()
