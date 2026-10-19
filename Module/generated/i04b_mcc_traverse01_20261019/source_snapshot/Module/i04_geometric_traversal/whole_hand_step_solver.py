"""Warm active-set solve of the unchanged bounded damped IK objective.

Every result is checked against the original box and KKT conditions. Numerical
failure falls back to the original BVLS implementation, never to clipping an
unconstrained IK command. The plant and independent guards do not change.
"""
import numpy as np
from .belly_bounded_step import bounded_belly_step


class WholeHandStepSolver:
    def __init__(self):
        self.active=None
        self.previous_delta=None
        self.fallbacks=0

    def solve(self,J,error,prior,lower,upper,*,gain=.55,damping=.001,maximum_joint_step=.0008,maximum_iterations=100):
        J=np.asarray(J,float);error=np.asarray(error,float);prior=np.asarray(prior,float)
        lower=np.asarray(lower,float);upper=np.asarray(upper,float);n=len(prior)
        step=np.broadcast_to(maximum_joint_step,(n,))
        def fallback():
            self.fallbacks+=1;self.active=None;self.previous_delta=None
            return bounded_belly_step(J,error,prior,lower,upper,gain=gain,damping=damping,
                maximum_joint_step=maximum_joint_step,maximum_iterations=maximum_iterations)
        if (J.ndim!=2 or J.shape[1]!=n or error.shape!=(J.shape[0],) or lower.shape!=(n,) or upper.shape!=(n,)
                or not all(np.isfinite(x).all() for x in (J,error,prior,lower,upper,step))
                or np.any(prior<lower) or np.any(prior>upper) or np.any(step<=0) or not 0<gain<=1 or damping<=0):return fallback()
        lo=np.maximum(-step,lower-prior);hi=np.minimum(step,upper-prior);fixed=lo==hi
        reserve=np.minimum((hi-lo)*.25,16*np.finfo(float).eps*np.maximum(1.,abs(prior)))
        sl,sh=lo+reserve,hi-reserve
        H=J.T@J+damping*damping*np.eye(n);b=-gain*(J.T@error)
        active=np.zeros(n,int) if self.active is None or len(self.active)!=n else self.active.copy()
        x=np.zeros(n) if self.previous_delta is None or len(self.previous_delta)!=n else self.previous_delta.copy()
        x=np.clip(x,sl,sh);active[fixed]=2;x[fixed]=lo[fixed]
        x[active==-1]=sl[active==-1];x[active==1]=sh[active==1]
        try:
            for iteration in range(maximum_iterations):
                free=active==0;bound=~free;trial=x.copy()
                if free.any():trial[free]=np.linalg.solve(H[np.ix_(free,free)],b[free]-H[np.ix_(free,bound)]@x[bound])
                delta=trial-x;scale=1.;hit=-1;side=0
                for i in np.flatnonzero(free):
                    if trial[i]<sl[i]:candidate=(sl[i]-x[i])/delta[i];edge=-1
                    elif trial[i]>sh[i]:candidate=(sh[i]-x[i])/delta[i];edge=1
                    else:continue
                    if candidate<scale:scale=max(0.,float(candidate));hit=int(i);side=edge
                if hit>=0:
                    x+=scale*delta;active[hit]=side;x[hit]=sl[hit] if side==-1 else sh[hit]
                    continue
                x=trial;gradient=H@x-b
                violation=np.zeros(n);violation[active==-1]=-gradient[active==-1];violation[active==1]=gradient[active==1]
                worst=int(np.argmax(violation))
                if violation[worst]<=1e-12:
                    # Evaluate KKT in the original augmented least-squares
                    # formulation as well, avoiding a normal-equation-only test.
                    grad=J.T@(J@x+gain*error)+damping*damping*x
                    residual=np.where(active==0,abs(grad),np.where(active==-1,np.maximum(-grad,0.),np.where(active==1,np.maximum(grad,0.),0.)))
                    command=prior+x
                    if (not np.isfinite(x).all() or residual.max()>1e-12 or np.any(x<lo) or np.any(x>hi)
                            or np.any(command<lower) or np.any(command>upper) or np.any(abs(x)>step)
                            or np.linalg.norm(error+J@x)>np.linalg.norm(error)+1e-12):return fallback()
                    self.active=active;self.previous_delta=x.copy()
                    return x,dict(feasible=True,reason='WARM_BOUNDED_SHARED_DLS',optimality_confirmed=True,
                        kkt_max=float(residual.max()),iterations=iteration+1,authority='LINEAR_GEOMETRY_STEP_ONLY_NO_CONTACT_OR_COVERAGE')
                active[worst]=0
            return fallback()
        except (np.linalg.LinAlgError,FloatingPointError,ValueError):return fallback()
