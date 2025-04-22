from __future__ import annotations
from numiphy.odesolvers.ode import *
from numiphy.symlib.symcore import *
from numiphy.toolkit.plotting import *


class VariationalBohmianSystem(OdeSystem):

    psi: Expr
    DELTA_T: float

    def __new__(cls, arg1: Expr|tuple[Expr, Expr, Expr], t: Symbol, x: Symbol, y: Symbol, delx: Symbol, dely: Symbol, args: Iterable[Symbol], DELTA_T: float):        
        if hasattr(arg1, '__iter__'):
            psi, xdot, ydot, delx_dot, dely_dot = arg1
        else:
            psi = arg1
            xdot = Imag(psi.diff(x)/psi)
            ydot = Imag(psi.diff(y)/psi)
            delx_dot = xdot.diff(x)*delx + xdot.diff(y)*dely
            dely_dot = ydot.diff(x)*delx + ydot.diff(y)*dely
        

        renorm = SymbolicPeriodicEvent("Checkpoint", DELTA_T, DELTA_T, [x, y, delx/sqrt(delx**2 + dely**2), dely/sqrt(delx**2 + dely**2)], hide_mask=True)

        obj = object.__new__(cls)
        obj.psi = psi
        obj.DELTA_T = DELTA_T
        return cls._process_args(obj, [xdot, ydot, delx_dot, dely_dot], t, x, y, delx, dely, args=args, events=[renorm])

    def get_orbit(self, x0, y0, rtol=0, atol=1e-9, min_step=0, max_step=np.inf, first_step=0, args=())->VariationalBohmianOrbit:
        q0 = np.array([x0, y0, np.sqrt(2)/2, np.sqrt(2)/2])
        ode = self.get(0, q0, rtol, atol, min_step, max_step, first_step, args, method="RK45", no_math_errno=True)
        return VariationalBohmianOrbit(ode)

    def __eq__(self, other):
        if other is self:
            return True
        elif type(other) is type(self):
            return (self.psi, self.DELTA_T) == (other.psi, other.DELTA_T) and OdeSystem.__eq__(self, other)
        else:
            return False


class BohmianOrbit(Orbit):

    @property
    def x(self):
        return self.q[:, 0]
    
    @property
    def y(self):
        return self.q[:, 1]
    
    def figure(self, **kwargs):
        fig = SquareFigure("Bohmian Orbit", xlabel='x', ylabel='y')
        fig.add(LinePlot(x=self.x, y=self.y, **kwargs))
        return fig



class VariationalBohmianOrbit(BohmianOrbit):


    @property
    def delx(self):
        return self.q[:, 2]
    
    @property
    def dely(self):
        return self.q[:, 3]
    
    @property
    def checkpoints(self):
        return self.event_map["Checkpoint"]

    @property
    def logksi(self):
        ksi_array = np.linalg.norm(self.q[self.checkpoints, 2:], axis=1)
        logksi = np.cumsum(np.log(ksi_array))
        return logksi
    
    @property
    def t_lyap(self):
        return self.t[self.checkpoints]
    
    @property
    def lyap(self):
        t = self.t_lyap
        return self.logksi/t
    


class OrbitCollection:

    def __init__(self, model: VariationalBohmianSystem, ics: Iterable[tuple[float, float]], rtol=0, atol=1e-9, min_step=0, max_step=np.inf, first_step=0, args=()):
        self.model = model
        self.orbits = [self.model.get_orbit(*ic, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args) for ic in ics]
        self.t_current = 0

    @property
    def DELTA_T(self):
        return self.model.DELTA_T

    @property
    def good_orbits(self):
        return [orb for orb in self.orbits if not orb.is_dead]
    
    def integrate(self, interval, num, max_frames=0, threads=-1):  
        goal = self.t_current+interval
        delta_t = num*self.DELTA_T
        N = round(interval//delta_t)+1
        for i in range(N):
            print(i, '/', N)
            if not self.good_orbits:
                break
            t_int = min(delta_t, goal-self.t_current)
            self.model.integrate_all(self.good_orbits, interval=t_int, max_frames=max_frames, threads=threads, max_events=-1)
            self.t_current += t_int
            if t_int < self.DELTA_T:
                break
    
    @property
    def t(self):
        return np.arange(self.DELTA_T, (self.t_current//self.DELTA_T)*self.DELTA_T + 1, self.DELTA_T)

    @property
    def lyap_all(self):
        return np.array([orb.lyap for orb in self.good_orbits])
    
    @property
    def lyap_mean(self):
        return np.mean(self.lyap_all, axis=0)
    
    @property
    def lyap_std(self):
        return np.std(self.lyap_all, axis=0)
    
    def hist(self, i=-1, bins=10, range=None, density=None, weights=None):
        data = self.lyap_all
        hist_data, bin_edges = np.histogram(data[:, i], bins, range, density, weights)
        return bin_edges[1:]+np.diff(bin_edges)/2, hist_data