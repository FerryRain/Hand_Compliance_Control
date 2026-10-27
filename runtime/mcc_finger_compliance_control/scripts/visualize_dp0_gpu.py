"""Read-only native viewer on the exact DP0/DPPO GPU evaluation chain."""
import argparse,json,time
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import NativeMujocoViewer
import evaluate_dp0_closed_loop as evaluation
from batched_b_control_chain import BatchedBControlChain
from batched_fullhand_mcc import _quat_matrix

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'mcc_finger_compliance_control/data'


class RestartReplay(Exception):
    """Restart the complete controller and episode, not just the physics."""


class ContactDisplay:
    def __init__(self):
        self.chain=None;self.contacts=None;self.frame=0;self.settling=True
        self.active=True;self.q=None;self.pose=None;self.nominal=None;self.normal=None

    @torch.no_grad()
    def capture(self,chain,kwargs,diagnostic):
        self.chain=chain;ctx=chain.context
        self.frame=ctx['frame'];self.settling=ctx['settling'];self.active=bool(ctx['active'][0])
        self.q=kwargs['q'].clone();self.pose=kwargs['palm_pose_world'].clone()
        self.nominal=diagnostic['nominal_q'].clone();self.normal=chain.direction.previous.clone()

    @torch.no_grad()
    def draw(self,visualizer):
        if self.chain is None or self.contacts is None:return
        rotation=_quat_matrix(self.pose)
        def world(q):
            points=self.chain.mcc.fk(q)
            return (self.pose[:,None,:3]+torch.einsum('bij,bfj->bfi',rotation,points))[0].cpu().numpy()
        targets=world(self.nominal);tips=world(self.q);normal=self.normal[0].cpu().numpy()
        x={k:v[0].cpu().numpy() for k,v in self.contacts.items()}
        colors=((1.,.3,.2,.95),(.2,.7,1.,.95),(1.,.8,.15,.95),(.8,.3,1.,.95))
        for finger,color in enumerate(colors):
            if np.isfinite(targets[finger]).all():
                visualizer.add_sphere(targets[finger],radius=.005,color=color)
                visualizer.add_arrow(targets[finger],targets[finger]+.025*normal[finger],color=color,width=.0025)
            if x['pad_found'][finger]:
                point=x['pad_points'][finger]
                direction=normal[finger]
                if x.get('physical_normal_found',np.zeros(4,dtype=bool))[finger]:
                    point=x['physical_contact_point'][finger]
                    direction=-x['physical_normal_inward'][finger]
                if np.isfinite(point).all():
                    loaded=np.isfinite(x['pad_force'][finger]).all() and np.linalg.norm(x['pad_force'][finger])>=.05
                    contact_color=(.15,1.,.2,1.) if loaded else (1.,.5,.05,1.)
                    visualizer.add_sphere(point,radius=.006,color=contact_color)
                    if np.isfinite(direction).all():
                        visualizer.add_arrow(point,point+.02*direction,color=contact_color,width=.0015)
            elif np.isfinite(tips[finger]).all():
                visualizer.add_sphere(tips[finger],radius=.007,color=(1.,.05,.05,1.))
            if x['connector_found'][finger] and np.isfinite(x['connector_points'][finger]).all():
                visualizer.add_sphere(x['connector_points'][finger],radius=.004,color=(1.,.65,.05,1.))

    def labels(self,frames):
        if self.contacts is None:return 'Initializing', '--', '--', '--'
        x=self.contacts
        strict=x['pad_found']&torch.isfinite(x['pad_points']).all(-1)&torch.isfinite(x['pad_force']).all(-1)&(x['pad_force'].norm(dim=-1)>=.05)&self.active
        found=(x['pad_found']|x['connector_found'])&self.active
        flags=lambda a:'/'.join(str(int(v)) for v in a[0].cpu())
        stage='Settling' if self.settling else ('Running' if self.active else 'Safety stop')
        progress=stage if self.settling else f'{stage} {self.frame}/{frames-1}'
        return progress,flags(strict),flags(found),'/'.join(f'{v:.1f}' for v in x['pad_force'][0].norm(dim=-1).cpu())


class DrivenNativeViewer(NativeMujocoViewer):
    """Original native UI, with stepping owned by the accepted GPU replay loop."""
    def __init__(self,env,display,frames,frame_rate):
        super().__init__(env,lambda obs:None,frame_rate=frame_rate,enable_perturbations=False)
        self.display=display;self.frames=frames;self.single_requested=False
        self.next_step_time=0.;self.previous_step_time=None

    def _step_physics(self,dt):
        pass  # Replay advances exactly one step; never advance again from UI.

    def _single_step(self):
        if self._is_paused:self.single_requested=True

    def reset_environment(self):
        raise RestartReplay

    def setup(self):
        super().setup()
        now=time.perf_counter();self._stats_last_time=now;self._last_tick_time=now

    def _update_stats(self):
        # Render FPS remains meaningful while paused; control rate becomes0.
        now=time.perf_counter();elapsed=now-self._stats_last_time
        if elapsed>=.5:
            self._fps=self._stats_frames/elapsed;self._sps=self._stats_steps/elapsed
            self._stats_frames=0;self._stats_steps=0;self._stats_last_time=now

    def _set_status_overlay(self,handle):
        status=self.get_status();stage,strict,found,force=self.display.labels(self.frames)
        font=mujoco.mjtFontScale.mjFONTSCALE_150.value
        handle.set_texts([
            (font,mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
             'Control step\nStatus\nRender FPS\nControl steps/s\nTarget RT\nActual RT',
             f'{status.step_count}\n'+('PAUSED' if status.paused else 'RUNNING')+
             f'\n{self._fps:.1f}\n{self._sps:.1f}\n{status.target_realtime:.2f}x\n{status.actual_realtime:.2f}x'),
            (font,mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
             'Trajectory\nStrict I/M/R/T\nFound I/M/R/T\nPad force N\nMarkers',
             f'{stage}\n{strict}\n{found}\n{force}\nGreen:tip  Amber:connector  Red:loss')])

    def before_step(self):
        while self.is_running():
            self.tick();self._update_stats()
            if self.single_requested or (not self._is_paused and time.perf_counter()>=self.next_step_time):
                self.single_requested=False;return
            time.sleep(.001)
        raise KeyboardInterrupt

    def after_step(self):
        now=time.perf_counter();period=self.env.unwrapped.step_dt/self._time_multiplier
        self._was_capped=self.previous_step_time is not None and now-self.previous_step_time>period+.002
        self.previous_step_time=now;self.next_step_time=now+period
        self._step_count+=1;self._stats_steps+=1
        if not self.is_running():raise KeyboardInterrupt
        self.tick();self._update_stats()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-id',type=int,default=21)
    parser.add_argument('--checkpoint',type=Path,default=DATA/'DPPO/DP0_zero_residual_canary_20261012/candidate.pt')
    parser.add_argument('--mcc-profile',type=Path,default=DATA/'contact_projection_full_development_20261011/selected_mcc_profile.json')
    parser.add_argument('--frames',type=int,default=1500)
    parser.add_argument('--seed',type=int,default=20261016)
    parser.add_argument('--render-fps',type=float,default=60.)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if not 76<args.frames<=1500:raise ValueError('frames must be77..1500')
    if args.render_fps<=0:raise ValueError('render-fps must be positive')
    checkpoint=args.checkpoint.resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    split=json.loads((evaluation.JOB/'DP0_split.json').read_text())
    if args.episode_id not in {x['episode_id'] for x in split['episodes']}:raise ValueError('Episode is outside current221 data')
    profile=json.loads(args.mcc_profile.read_text())
    scale=evaluation.scheduler_scale()
    evaluation.DP0_MODEL=checkpoint
    original_step=ManagerBasedRlEnv.step
    original_close=ManagerBasedRlEnv.close
    original_chain_step=BatchedBControlChain.step
    original_observer=evaluation.ContactSlotObserver
    display=ContactDisplay();viewer=None
    class DisplayObserver(original_observer):
        def read(self,*args,**kwargs):
            result=super().read(*args,**kwargs)
            display.contacts={k:v.clone() for k,v in self.latest.items()}
            return result
    def watched_chain_step(chain,**kwargs):
        action,diagnostic=original_chain_step(chain,**kwargs)
        display.capture(chain,kwargs,diagnostic)
        return action,diagnostic
    def displayed_step(env,action):
        nonlocal viewer
        if viewer is None:
            cfg=env.cfg.viewer
            cfg.distance=.45;cfg.azimuth=45.;cfg.elevation=-10.
            cfg.origin_type=cfg.OriginType.ASSET_BODY
            cfg.entity_name='robot';cfg.body_name='palm_lower'
            base_visualizers=env.update_visualizers
            def contact_visualizers(visualizer):
                base_visualizers(visualizer);display.draw(visualizer)
            env.update_visualizers=contact_visualizers
            viewer=DrivenNativeViewer(env,display,args.frames,args.render_fps)
            viewer.setup()
        viewer.before_step()
        result=original_step(env,action)
        viewer.after_step()
        return result
    def displayed_close(env):
        nonlocal viewer
        if viewer is not None:viewer.close();viewer=None
        return original_close(env)
    ManagerBasedRlEnv.step=displayed_step
    ManagerBasedRlEnv.close=displayed_close
    BatchedBControlChain.step=watched_chain_step
    evaluation.ContactSlotObserver=DisplayObserver
    print(f'[VISUAL] checkpoint={checkpoint}; ep={args.episode_id}; frames={args.frames};'
          ' CUDA physics/MCC,20 DDPM steps,zero residual,100 settle,100N stop',flush=True)
    print('[VISUAL] Space:pause  Right:single step  +/-:speed  Enter:restart  R:markers. Close/Ctrl-C:exit.',flush=True)
    try:
        while True:
            try:
                report=evaluation.evaluate([args.episode_id],args.frames,args.seed,'cuda:0',
                    contact_gate='whole_tip',inference_steps=20,mcc_profile=profile,scheduler_scale_override=scale)
                break
            except RestartReplay:
                display=ContactDisplay()
                print('[VISUAL] Restarting the complete episode/controller.',flush=True)
            # Leave the exception handler before collecting: its traceback
            # otherwise retains the old environment and CUDA closure cycles.
            import gc
            gc.collect();torch.cuda.empty_cache()
        report['viewer']='Read-only native CPU rendering mirror; physics/control remain CUDA'
        if args.report is not None:evaluation.write(args.report,report)
        lane=report['lanes'][0]
        print(json.dumps(dict(episode_id=args.episode_id,safety_stop_frame=lane['safety_stop_frame'],
            strict_healthy=lane['post_bootstrap']['strict_healthy_four_rate'],
            found=lane['post_bootstrap']['found_four_rate'],peak_force_n=lane['peak_force_n']),indent=2))
    except KeyboardInterrupt:print('[VISUAL] Window closed / interrupted.',flush=True)
    finally:
        ManagerBasedRlEnv.step=original_step;ManagerBasedRlEnv.close=original_close
        BatchedBControlChain.step=original_chain_step;evaluation.ContactSlotObserver=original_observer
        if viewer is not None:viewer.close()


if __name__=='__main__':main()
