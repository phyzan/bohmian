from __future__ import annotations
from numiphy.symlib.symcore import *
from numiphy.toolkit.plotting import *
from odepack import *
import mpmath


class BohmianOrbit(LowLevelODE):

    @property
    def x(self):
        return self.q[:, 0]
    
    @property
    def y(self):
        return self.q[:, 1]

class VariationalBohmianOrbit(VariationalLowLevelODE):
    
    @property
    def x(self):
        return self.q[:, 0]
    
    @property
    def y(self):
        return self.q[:, 1]
    
    @property
    def delx(self):
        return self.q[:, 2]
    
    @property
    def dely(self):
        return self.q[:, 3]


class VariationalBohmianSystem(OdeSystem):

    psi: Expr
    DELTA_T: float

    def __init__(self, arg1: Expr|tuple[Expr, Expr, Expr, Expr, Expr], t: Symbol, x: Symbol, y: Symbol, delx: Symbol, dely: Symbol, args: Iterable[Symbol], DELTA_T: float, module_name: str=None, directory: str = None):
        if hasattr(arg1, '__iter__'):
            psi, xdot, ydot, delx_dot, dely_dot = arg1
        else:
            psi = arg1
            xdot = Imag(psi.diff(x)/psi)
            ydot = Imag(psi.diff(y)/psi)
            delx_dot = xdot.diff(x)*delx + xdot.diff(y)*dely
            dely_dot = ydot.diff(x)*delx + ydot.diff(y)*dely
        
        self.psi = psi
        self.DELTA_T = DELTA_T
        OdeSystem.__init__(self, [xdot, ydot, delx_dot, dely_dot], t, [x, y, delx, dely], args=args, directory=directory, module_name=module_name)

    def get_orbit(self, x0, y0, t0=0., rtol=0., atol=1e-9, min_step=0, max_step=np.inf, first_step=0, args=(), method="RK45", dtype='double'):
        return VariationalBohmianOrbit(f=self.lowlevel_odefunc(dtype), jac=self.lowlevel_jac(dtype), t0=t0, q0=[x0, y0, 1, 1], period=self.DELTA_T, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, dtype=dtype)

    def __eq__(self, other):
        if other is self:
            return True
        elif type(other) is type(self):
            return (self.psi, self.DELTA_T) == (other.psi, other.DELTA_T) and OdeSystem.__eq__(self, other)
        else:
            return False

