from __future__ import annotations
from abc import ABC, abstractmethod
import math
from numiphy.findiffs import grids
from numiphy.symlib.geom import Line2D, Parallelogram
from numiphy.symlib.vectorfields import VectorField2D
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sciopt
from numiphy.odesolvers import *
from numiphy.symlib.pylambda import ScalarLambdaExpr, VectorLambdaExpr
from .orbits import *


class Hermite:

    def __init__(self, *x: Symbol):
        self.x = x
        self.nd = len(x)

    def __call__(self, *n: int):
        polys = []
        for i in range(self.nd):
            polys.append(HermitePoly(n[i], self.x[i]))
        return Mul(*polys)


class SolvedPotential(ABC):

    weight: Expr

    def __init__(self, x: tuple[Symbol], m=1.):
        self.x = x
        self.nd = len(x)
        self.t = Symbol('t')
        self.m = m

    @abstractmethod
    def phi(*n)->QuantumState:
        pass

    @abstractmethod
    def Phi(*n, weight=True)->Expr:
        pass

    @abstractmethod
    def energy(self, *n)->float:
        pass
        

class QuantumOscillator(SolvedPotential):
    '''
    hbar = 1 is assumed
    '''
    def __init__(self, x: tuple[Symbol], omega: tuple[float]=None, m=1.):
        super().__init__(x, m)
        if omega is None:
            omega = tuple(self.nd*[1])
        self.omega = omega
        self.x0 = tuple([math.sqrt(1/(m*omega[i])) for i in range(self.nd)])
        self.weight = exp(-Add(*[(self.x[i]/self.x0[i])**2/2 for i in range(self.nd)]))

    def phi(self, *n):
        return QuantumState(self, n, 1)
    
    def Phi(self, *n: int, weight=True):
        coef = 1
        polys = []
        for i in range(self.nd):
            coef *= 1/math.sqrt(2**n[i]*math.factorial(n[i]))*(math.pi*self.x0[i]**2)**-0.25
            polys.append(HermitePoly(n[i], self.x[i]).replace({self.x[i]: self.x[i]/self.x0[i]}))
        if weight:
            return Mul(coef, *polys, self.weight)
        else:
            return Mul(coef, *polys)
    
    def energy(self, *n):
        return sum([self.omega[i]*(n[i]+0.5) for i in range(self.nd)])


class WaveFunction:

    def __init__(self, qs: list[QuantumState]):
        self.V = qs[0].V
        self.qs = tuple(qs)
        self.norm_coef = math.sqrt(sum([abs(qsi.coef)**2 for qsi in qs]))

    def __repr__(self):
        return str(self)
    
    def __str__(self):
        return ' + '.join([str(q) for q in self.qs])

    def __add__(self, other: WaveFunction):
        return _WaveFunction(list(self._all_states()+other._all_states()))
    
    def __sub__(self, other: WaveFunction):
        return self + -other
    
    def __neg__(self):
        return -1*self
    
    def __mul__(self, coef):
        assert coef != 0
        return _WaveFunction([coef*qsi for qsi in self._all_states()])
    
    def __rmul__(self, coef)->WaveFunction:
        return self * coef
    
    def __truediv__(self, num):
        return 1/num * self

    def _all_states(self):
        return self.qs
    
    def get(self, time=False, norm=True):
        return Add(*[fi.get_weightfree(time=time) for fi in self.qs]).expand()*self.V.weight/self.norm_coef**norm
    
    def ascallable(self, time=False, norm=True):
        psi = self.get(time=time, norm=norm)
        return ScalarLambdaExpr(psi, *self.V.x, self.V.t)


class QuantumState(WaveFunction):


    def __init__(self, V: SolvedPotential, n: tuple[int], coef: float):
        if coef == 0:
            raise ValueError('coef must be != 0')
        self.V = V
        self.n = n
        self.coef = coef

    def __mul__(self, coef):
        assert coef != 0
        return QuantumState(self.V, self.n, coef*self.coef)

    def _all_states(self):
        return self,

    def __repr__(self):
        return f'{self.coef}*|{", ".join([str(i) for i in self.n])}>'
    
    def __str__(self):
        return repr(self)

    def get(self, time=False, norm=True):
        psi = self.V.Phi(*self.n)
        if not norm:
            psi = self.coef*psi
        if time is False:
            return psi
        else:
            return psi * exp(-1j*self.V.energy(*self.n)*self.V.t)

    def get_weightfree(self, time=False):
        psi = self.coef*self.V.Phi(*self.n, weight=False)
        if time is False:
            return psi
        else:
            return psi * exp(-1j*self.V.energy(*self.n)*self.V.t)


def _WaveFunction(qs: list[QuantumState]):
    for q in qs:
        if q.V is not qs[0].V:
            raise NotImplementedError('Can only create a Wavefunction comprised of steady states of the same Hamiltonian')
    V = qs[0].V
    i = 0
    while i < len(qs):
        if len(qs) < 2:
            break
        j = i+1
        while j < len(qs):
            if qs[j].n == qs[i].n:
                coef = qs[i].coef + qs[j].coef
                if coef != 0:
                    qs[i] = QuantumState(V, qs[i].n, coef)
                    qs.pop(j)
                else:
                    qs.pop(j)
                    qs.pop(i)
                    break
            else:
                j += 1
        if j == len(qs):
            i += 1
    
    if not qs:
        raise ValueError('The Wavefunction is 0')
    elif len(qs) == 1:
        return qs[0]
    else:
        return WaveFunction(qs)


def HermitePoly(n: int, x: Symbol)->Expr:
    '''
    Hermite Polynomial of the n-th order.

    This can be calculated using two equivalent methods:

    1) H_n(x) = (-1)^n * exp(x**2) * d^n/dx^n exp(-x^2)
    2) H_n(x) = (2*x - d/dx)^n * 1

    They can also be obtained using the recursive formula

    H_{n+1}(x) = 2*x*H_n(x) - 2*n*H_{n-1}(x)
    '''

    if n == 0:
        return S.One
    elif n == 1:
        return 2*x
    else:
        h0, h1 = S.One, 2*x
        for i in range(2, n+1):
            h2 = 2*x*h1 - 2*(i-1)*h0
            h0, h1 = h1, h2
        return h2.expand()
    

class Bohmian2D(VectorField2D):


    '''
    In Transformed bohmian class, keep only imag.
    Keep true field and complex field separate, no need
    for real=True. Maybe do this in Transformed too, keep both but separate.


    Transformed field creates autonomos ode's of u, v
    with fictitious time parameter "s", not t
    '''

    def __init__(self, data: Expr | tuple[Expr, ...], symbols: tuple[Symbol, ...], args: tuple[Symbol, ...] = (), box=[-10., 10., -10, 10.], rmax=1e-2, Nsub=100):
        var_data: tuple[Expr, ...] = None
        if hasattr(data, '__iter__'):
            data: tuple[Expr, ...] = data
            if len(data) == 3:
                x, y, t = symbols
                psi, xdot, ydot = data
            elif len(data) == 5:
                x, y, delx, dely, t = symbols
                psi, xdot, ydot, delxdot, delydot = data
                var_data = (delx, dely, delxdot, delydot)
            else:
                raise ValueError("")

        else:
            psi: Expr = data
            x, y, t = symbols
            xdot, ydot = Imag(psi.diff(x)/psi), Imag(psi.diff(y)/psi)

        if var_data is None:
            delx, dely = Symbol(f'del_{x.name}'), Symbol(f'del_{y.name}')
            delxdot = xdot.diff(x)*delx + xdot.diff(y)*dely
            delydot = ydot.diff(x)*delx + ydot.diff(y)*dely

        self.tvar, self.xvar, self.yvar, self.delx, self.dely = t, x, y, delx, dely
        self._odesys_data = xdot, ydot
        self._varodesys_data = xdot, ydot, delxdot, delydot
        self.Psi = ScalarLambdaExpr(psi, x, y, self.tvar)
        self.gradPsi = VectorLambdaExpr([psi.diff(x), psi.diff(y)], x, y, self.tvar)

        self.box = box
        self.rmax = rmax
        self.Nsub=Nsub
        self.args = args

        VectorField2D.__init__(self, xdot, ydot, x, y, self.tvar)

    @property
    def psi(self)->Expr:
        return self.Psi.expr
    
    @cached_property
    def quantum_potential(self):
        psi = self.psi
        R = sqrt(Imag(psi)**2 + Real(psi)**2)
        LR = R.diff(self.xvar, 2) + R.diff(self.yvar, 2)
        return -LR/(2*R)

    def jacPsi(self, q, t, *args):
        '''
        Construct the Jacobian of the wavefunction as a python callable.
        The jacobian is to be passed in a numerical solver to solve the system of
        equations
        Psi_R(x, y) = 0
        Psi_Im(x, y) = 0

        This is J = [[Re(Ψx), Re(Ψy)],
                     [Im(Ψx), Im(Ψy)]
        '''
        return [self.gradPsi(*q, t, *args).real, self.gradPsi(*q, t, *args).imag]

    def _psi_eqsystem(self, q, t, *args):
        psi = self.Psi(*q, t, *args)
        return [psi.real, psi.imag]

    def node(self, t, *args)->np.ndarray:
        sq = Parallelogram(*self.box)
        flow = self.flow(sq, t)
        if abs(flow) < 1e-4:
            raise ValueError('No indication of vortex within the search limits of the Vector Field.')
        else:
            for _ in range(self.Nsub):
                sqrs = sq.split()

                for i in range(4):
                    sq = sqrs[i]
                    flow = self.flow(sq, t)
                    if abs(flow) > 1e-4:
                        ri = 0.5*(sq.Lx**2 + sq.Ly**2)**0.5
                        if ri <= self.rmax:
                            return sciopt.root(self._psi_eqsystem, sq.center, args=(t, *args), jac=self.jacPsi).x
                        else:
                            break
    
    def _grid(self, arrows: int):
        return grids.Uniform1D(*self.box[:2], arrows) * grids.Uniform1D(*self.box[2:], arrows)

    def xNdot(self, t, dt=1e-3)->list[float|complex]:
        s1 = self.node(t-dt)
        s2 = self.node(t+dt)
        
        xNdot, yNdot = (s2-s1)/(2*dt)
        return [xNdot, yNdot]
    
    def flow(self, line: Line2D, t, *args):
        return super().flow(line, t, *args)

    def loop(self, q, r, t, *args):
        return super().loop(q, r, t, *args)

    def plot(self, t, arrows=30, *args, **kwargs):
        return super().plot(self._grid(arrows), t, *args, **kwargs)
    
    def plot_line(self, line: Line2D, t, arrows=30, n=400, *args, **kwargs):
        return super().plot_line(line, self._grid(arrows), n, t, *args, **kwargs)
    
    def plot_circle(self, q, r, t, arrows=30, n=400, *args, **kwargs):
        return super().plot_circle(q, r, self._grid(arrows), n, t, *args, **kwargs)

    def transformed(self, t, dt=1e-3):
        xN, yN = self.node(t)
        xNdot, yNdot = self.xNdot(t, dt)
        u, v = variables('u, v')
        Udot = self.x.expr.subs({self.xvar: u+xN, self.yvar: v+yN, self.tvar: t}) - xNdot*1j
        Vdot = self.y.expr.subs({self.xvar: u+xN, self.yvar: v+yN, self.tvar: t}) - yNdot*1j
        return VectorField2D(Imag(Udot), Imag(Vdot), u, v, *self.args)

    def Xpoint_linedata(self, t, s, u0=-1., v0=-1., dt=1e-3, h=1e-5, args=(), **odeargs):
        f = self.transformed(t, dt, *args)
        u, v = f.fixed_point(u0, v0)
        p = np.array([u, v])
        jac = f.Jac(u, v)
        (lb, lr), (vecb, vecr) = np.linalg.eigh(jac)
        assert lb < 0 and lr > 0
        xb1, yb1 = f.streamline(*(p+h*vecb), s, **odeargs)
        xb2, yb2 = f.streamline(*(p-h*vecb), s, **odeargs)

        xr1, yr1 = f.streamline(*(p+h*vecr), -s, **odeargs)
        xr2, yr2 = f.streamline(*(p-h*vecr), -s, **odeargs)

        return f, xb1, yb1, xb2, yb2, xr1, yr1, xr2, yr2

    def plot_Xpoint(self, grid: grids.Grid, t, s, u0=-1., v0=-1., dt=1e-3, h=1e-5, args=(), **odeargs):
        f, xb1, yb1, xb2, yb2, xr1, yr1, xr2, yr2 = self.Xpoint_linedata(t, s, u0, v0, dt, h, args, **odeargs)
        fig, ax = f.plot(grid)
        ax.plot(xb1, yb1, color='blue')
        ax.plot(xb2, yb2, color='blue')
        ax.plot(xr1, yr1, color='red')
        ax.plot(xr2, yr2, color='red')
        return fig, ax
    
    @cached_property
    def ode_system(self):
        return OdeSystem(self._odesys_data, self.tvar, self.xvar, self.yvar, args=self.args)
    
    def varode_sys(self, DELTA_T):
        return VariationalBohmianSystem((self.psi, *self._varodesys_data), self.tvar, self.xvar, self.yvar, self.delx, self.dely, self.args, DELTA_T)
    
    def orbit(self, x0, y0, t0=0., rtol=0, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=()):
        ode = self.ode_system.get(t0, np.array([x0, y0]), rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method="RK45", no_math_errno=True)
        return BohmianOrbit(ode)
    
    def variational_orbit(self, x0, y0, t0=0., rtol=0, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), DELTA_T=0.05):
        return self.varode_sys(DELTA_T=DELTA_T).get_orbit(x0, y0, t0=t0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args)
    

