from __future__ import annotations
from abc import ABC, abstractmethod
import math
from numiphy.findiffs import grids
from numiphy.symlib.geom import Line2D, Parallelogram
from odepack import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sciopt
from scipy.integrate import IntegrationWarning
import warnings
from numiphy.symlib.pylambda import ScalarLambdaExpr, VectorLambdaExpr
from .orbits import *


class SolvedPotential(ABC):

    weight: Expr

    def __init__(self, x: tuple[Symbol], t: Symbol, m=1, hbar=1):
        self.x = x
        self.nd = len(x)
        self.t = t
        self.m = m
        self.hbar = hbar

    @abstractmethod
    def phi(*n, weight=True)->Expr:
        pass

    @abstractmethod
    def energy(self, *n)->float:
        pass

    def Phi(self, *n):
        return self.phi(*n) * exp(-1j*self.energy(*n)*self.t)
        

class QuantumOscillator(SolvedPotential):

    def __init__(self, x: tuple[Symbol], t: Symbol, omega: tuple[Symbol], m=1, hbar=1):
        super().__init__(x, t, m, hbar)
        self.omega = omega
        self.x0 = tuple([sqrt(self.hbar/(self.m*omega[i])) for i in range(self.nd)])
        self.weight = exp(-Add(*[(self.x[i]/self.x0[i])**2/2 for i in range(self.nd)]))
    
    def phi(self, *n: int, weight=True):
        coef = 1
        polys = []
        for i in range(self.nd):
            coef *= 1/sqrt(2**n[i]*math.factorial(n[i]))*(S.pi*self.x0[i]**2)**Rational(-1, 4)
            polys.append(HermitePoly(n[i], self.x[i]/self.x0[i]))
        if weight:
            return Mul(coef, *polys, self.weight)
        else:
            return Mul(coef, *polys)
    
    def energy(self, *n):
        return sum([self.hbar*self.omega[i]*(n[i]+Rational(1, 2)) for i in range(self.nd)])



def HermitePoly(n: int, x: Expr)->Expr:
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
    

class Bohmian2D:


    '''
    In Transformed bohmian class, keep only imag.
    Keep true field and complex field separate, no need
    for real=True. Maybe do this in Transformed too, keep both but separate.


    Transformed field creates autonomos ode's of u, v
    with fictitious time parameter "s", not t
    '''

    def __init__(self, data: Expr | tuple[Expr, ...], V: Expr, symbols: tuple[Symbol, ...], args: tuple[Symbol, ...] = ()):
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

        self.bohm_field = ConservativeVectorField2D((xdot, ydot), x, y, self.tvar, *args)
        self._V = V

    @property
    def args(self):
        return self.bohm_field.args[1:]
    
    @property
    def psi(self)->Expr:
        return self.Psi.expr
    
    @property
    def rho(self)->Expr:
        return Abs(self.psi)**2
    
    @property
    def potential(self):
        return self._V
    
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

    def node(self, t, args, Nsub = 100, rmax=1e-2, box = [-10, 10, -10, 10])->np.ndarray:
        sq = Parallelogram(*box)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=IntegrationWarning)
            flow = self.flow(sq, t)

        error = lambda flow: ArithmeticError(f'No indication of vortex within the search limits of the Vector Field at t = {t}. Flow is {flow}')
        if abs(flow)>1e-3:
            for _ in range(Nsub):
                sqrs = sq.split()
                for i in range(4):
                    sq = sqrs[i]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=IntegrationWarning)
                        flow = self.flow(sq, t, args)
                    if abs(flow) > 1e-4:
                        ri = 0.5*(sq.Lx**2 + sq.Ly**2)**0.5
                        if ri <= rmax:
                            xn, yn = sciopt.root(self._psi_eqsystem, sq.center, args=(t, *args), jac=self.jacPsi).x
                            return np.array([xn, yn])
                        else:
                            break
                    elif i==3:
                        raise error(flow)
        else:
            raise error(flow)

    def xNdot(self, t, args=(), dt=1e-3, **kwargs):
        s1 = self.node(t-dt, args, **kwargs)
        s2 = self.node(t+dt, args, **kwargs)
        
        xNdot, yNdot = (s2-s1)/(2*dt)
        return np.array([xNdot, yNdot])
    
    def flow(self, line: Line2D, t, *args):
        return self.bohm_field.flow(line, t, *args)

    def loop(self, q, r, t, *args):
        return self.bohm_field.loop(q, r, t, *args)

    def plot(self, grid: grids.Grid, t, args, scaled=False, **kwargs):
        return self.bohm_field.plot(grid, (t, *args), scaled=scaled, **kwargs)

    def Y_point(self, t, args, x0=0, y0=0):
        return self.bohm_field.fixed_point(x0, y0, t, args)
    
    def X_point(self, t, args=(), x0=0, y0=0, **kwargs):
        xndot = self.node(t, args, **kwargs)
        return self.field_point(xndot, t, args, x0, y0)
    
    def field_point(self, value, t, args=(), x0=0, y0=0):
        f = lambda q, *args: self.bohm_field.call(q, *args) - value
        return sciopt.root(f, [x0, y0], jac = self.bohm_field.calljac, args=(t, *args)).x
    
    def eigen_lines(self, t, x0, y0, s, epsilon=1e-6, safety_dist=1e-5, curve_length=True, rich=False, **kwargs):
        kwargs['args'] = (t,) + kwargs.get('args', ())
        return self.bohm_field.eigen_lines(x0, y0, s, epsilon, safety_dist, curve_length, rich, **kwargs)

    def uv_field(self, t, args=(), dt=1e-3, **kwargs):
        to_sub = {self.args[i]: args[i] for i in range(len(self.args))}
        xN, yN = self.node(t, args, dt=dt, **kwargs)
        xNdot, yNdot = self.xNdot(t, args, dt=dt, **kwargs)
        u, v = variables('u, v')
        Udot = self.bohm_field.x.expr.subs(to_sub).subs({self.xvar: u+xN, self.yvar: v+yN, self.tvar: t}) - xNdot
        Vdot = self.bohm_field.y.expr.subs(to_sub).subs({self.xvar: u+xN, self.yvar: v+yN, self.tvar: t}) - yNdot
        return ConservativeVectorField2D([Udot, Vdot], u, v)
    
    @property
    def force_field(self):
        f = self._V + self.quantum_potential
        return ConservativeVectorField2D(f, self.xvar, self.yvar, self.tvar, *self.args)
    
    @cached_property
    def ode_system(self):
        return OdeSystem(self._odesys_data, self.tvar, [self.xvar, self.yvar], args=self.args)
    
    def varode_sys(self, DELTA_T):
        return VariationalBohmianSystem((self.psi, *self._varodesys_data), self.tvar, self.xvar, self.yvar, self.delx, self.dely, self.args, DELTA_T)
    
    def orbit(self, x0, y0, t0=0., rtol=0, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45"):
        s = self.ode_system
        return BohmianOrbit(LowLevelODE(s.lowlevel_odefunc, jac=s.lowlevel_jac, t0=t0, q0=np.array([x0, y0]), rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method))
    
    def variational_orbit(self, x0, y0, t0=0., rtol=0, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), DELTA_T=0.05):
        return self.varode_sys(DELTA_T=DELTA_T).get_orbit(x0, y0, t0=t0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args)
    
