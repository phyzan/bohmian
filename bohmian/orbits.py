from __future__ import annotations
from numiphy.symlib.symcore import *
from numiphy.toolkit.plotting import *
from odepack import *
from odepack.odesolvers_base import *
from odepack.symode import _get_cls_instance
from typing import TypeVar
import mpmath

T = TypeVar('T')

class _BohmianOrbit(AbstractLowLevelODE[T]):

    @property
    def x(self):
        return self.q[:, 0]
    
    @property
    def y(self):
        return self.q[:, 1]
    
class BohmianOrbit_Double(LowLevelODE_Double, _BohmianOrbit[np.float64]):
    pass

class BohmianOrbit_Float(LowLevelODE_Float, _BohmianOrbit[np.float32]):
    pass

class BohmianOrbit_LongDouble(LowLevelODE_LongDouble, _BohmianOrbit[np.longdouble]):
    pass

class BohmianOrbit_MpReal(LowLevelODE_MpReal, _BohmianOrbit[mpmath.mpf]):
    pass
    

class _VariationalBohmianOrbit(AbstractVariationalLowLevelODE[T]):
    
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


class VariationalBohmianOrbit_Double(VariationalLowLevelODE_Double, _VariationalBohmianOrbit[np.float64]):
    pass

class VariationalBohmianOrbit_Float(VariationalLowLevelODE_Float, _VariationalBohmianOrbit[np.float32]):
    pass

class VariationalBohmianOrbit_LongDouble(VariationalLowLevelODE_LongDouble, _VariationalBohmianOrbit[np.longdouble]):
    pass

class VariationalBohmianOrbit_MpReal(VariationalLowLevelODE_MpReal, _VariationalBohmianOrbit[mpmath.mpf]):
    pass


AnyBohmianOrbit: TypeAlias = Union[BohmianOrbit_Double, BohmianOrbit_Float, BohmianOrbit_LongDouble, BohmianOrbit_MpReal]
AnyVariationalBohmianOrbit: TypeAlias = Union[VariationalBohmianOrbit_Double, VariationalBohmianOrbit_Float, VariationalBohmianOrbit_LongDouble, VariationalBohmianOrbit_MpReal]


def BohmianOrbit(*args, dtype='double', **kwargs)->AnyBohmianOrbit:
    return _get_cls_instance('BohmianOrbit', dtype=dtype, *args, **kwargs)

def VariationalBohmianOrbit(*args, dtype='double', **kwargs)->AnyVariationalBohmianOrbit:
    return _get_cls_instance('VariationalBohmianOrbit', dtype=dtype, *args, **kwargs)


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

