from __future__ import annotations
from numiphy.symlib.symcore import *
from numiphy.toolkit.plotting import *
from odepack import *
import itertools


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

    def __new__(cls, arg1: Expr|tuple[Expr, Expr, Expr, Expr, Expr], t: Symbol, x: Symbol, y: Symbol, delx: Symbol, dely: Symbol, args: Iterable[Symbol], DELTA_T: float, module_name: str=None, directory: str = None)->VariationalBohmianSystem:
        if hasattr(arg1, '__iter__'):
            psi, xdot, ydot, delx_dot, dely_dot = arg1
        else:
            psi = arg1
            xdot = Imag(psi.diff(x)/psi)
            ydot = Imag(psi.diff(y)/psi)
            delx_dot = xdot.diff(x)*delx + xdot.diff(y)*dely
            dely_dot = ydot.diff(x)*delx + ydot.diff(y)*dely
        
        obj = CompileTemplate.__new__(cls, module_name=module_name, directory=directory)
        obj.psi = psi
        obj.DELTA_T = DELTA_T
        return cls._process_args(obj, [xdot, ydot, delx_dot, dely_dot], t, [x, y, delx, dely], args=args)

    def get_orbit(self, x0, y0, t0=0., rtol=0., atol=1e-9, min_step=0, max_step=np.inf, first_step=0, args=(), method="RK45"):
        return VariationalBohmianOrbit(f=self.lowlevel_odefunc, jac=self.lowlevel_jac, t0=t0, q0=[x0, y0, 1, 1], period=self.DELTA_T, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method)

    def __eq__(self, other):
        if other is self:
            return True
        elif type(other) is type(self):
            return (self.psi, self.DELTA_T) == (other.psi, other.DELTA_T) and OdeSystem.__eq__(self, other)
        else:
            return False


class OrbitCollection:

    def __init__(self, model: VariationalBohmianSystem, ics: Iterable[tuple[float, float]], args = (), **odeargs):
        self.model = model
        self._orbits = np.empty(shape = [len(x) if hasattr(x, '__iter__') else 1 for x in args], dtype=object)
        for ind, params in zip(np.ndindex(self._orbits.shape), itertools.product(*([arg if hasattr(arg, '__iter__') else [arg] for arg in args]))):
            self._orbits[*ind] = [self.model.get_orbit(*ic, args = tuple((params)), **odeargs) for ic in ics]

    def orbits(self, *index: int)->list[VariationalBohmianOrbit]:
        if not index:
            return self._orbits.flat[0]
        else:
            return self._orbits[*index]
        
    @property
    def all_orbits(self)->list[VariationalBohmianOrbit]:
        res = []
        for orbits in self._orbits.flat:
            res += orbits
        return res
    
    @property
    def DELTA_T(self):
        return self.model.DELTA_T
    
    def good_orbits(self, *index):
        return [orb for orb in self.orbits(*index) if not orb.is_dead]
    
    @property
    def all_good_orbits(self):
        return [orb for orb in self.all_orbits if not orb.is_dead]
    
    def integrate(self, interval, max_frames=-1, threads=-1, display_progress=False):
        integrate_all(self.all_orbits, interval=interval, max_frames=max_frames, threads=threads, display_progress=display_progress)
    
    @property
    def t_lyap(self):
        for orb in self.all_orbits:
            if not orb.is_dead:
                return orb.t_lyap
        raise ValueError('')
    
    @property
    def lyap_all(self):
        return np.array([orb.lyap for orb in self.all_good_orbits])
    
    
    def lyap(self, l=0.005, *index):
        res = np.array([orb.lyap for orb in self.good_orbits(*index)])
        return res[abs(res)[:, -1]>l]
    
    def lyap_mean(self, l=0.005, *index):
        return np.mean(self.lyap(l, *index), axis=0)
    
    def lyap_std(self, l=0.005, *index):
        return np.std(self.lyap(l, *index), axis=0)
    
    def lyap_mean_error(self, l=0.005, *index):
        sample = self.lyap(l, *index)
        std = np.std(sample, axis=0)
        return std/np.sqrt(sample.shape[0]-1)
    
    def hist(self, l=0.005, index=(), i=-1, bins=10, range=None, density=None, weights=None):
        data = self.lyap(l, *index)
        hist_data, bin_edges = np.histogram(data[:, i], bins, range, density, weights)
        return bin_edges[1:]+np.diff(bin_edges)/2, hist_data
    
    def data(self, l=0.005, *index):
        lyap = self.lyap(l, *index)
        err = self.lyap_mean_error(l, *index)
        return self.t_lyap, lyap, err