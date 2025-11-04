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
from rootfinder import *


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
            h2 = (2*x*h1 - 2*(i-1)*h0).expand()
            h0, h1 = h1, h2
        return h2
    

class Bohmian2D:


    '''
    In Transformed bohmian class, keep only imag.
    Keep true field and complex field separate, no need
    for real=True. Maybe do this in Transformed too, keep both but separate.


    Transformed field creates autonomos ode's of u, v
    with fictitious time parameter "s", not t
    '''

    def __init__(self, data: Expr | tuple[Expr, ...], V: Expr, symbols: tuple[Symbol, ...], args: tuple[Symbol, ...] = (), directory: str = None):
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
        self.Psi = ScalarLambdaExpr(psi, x, y, self.tvar, *args)
        self.gradPsi = VectorLambdaExpr([psi.diff(x), psi.diff(y)], x, y, self.tvar, *args)

        self.bohm_field = ConservativeVectorField2D((xdot, ydot), x, y, self.tvar, *args)
        self._V = V
        if directory is not None:
            self._odesys_modname = 'bohm_sys'
            self._varodesys_modname = 'var_bohm_sys'
            self._nodalpoint_modname = 'nodal_point_sys'
            self._xpoint_bin = 'xpoint_sys'
            self._dir = directory
        else:
            self._odesys_modname = None
            self._varodesys_modname = None
            self._nodalpoint_modname = None
            self._xpoint_bin = None
            self._dir = directory = None
    
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

    def xNdot(self, t, args=(), **kwargs):
        xn = self.node(t, args, **kwargs)
        return self.v_co(t, *xn, *args)
    
    def flow(self, line: Line2D, t, *args):
        return self.bohm_field.flow(line, t, *args)

    def loop(self, q, r, t, *args):
        return self.bohm_field.loop(q, r, t, *args)

    def plot(self, grid: grids.Grid, t, args, scaled=False, **kwargs):
        return self.bohm_field.plot(grid, (t, *args), scaled=scaled, **kwargs)

    def Y_point(self, t, args=(), x0=0, y0=0):
        return self.bohm_field.fixed_point(x0, y0, (t, *args))
    
    def X_point(self, t, args=(), x0=0, y0=0, **kwargs):
        return self.field_point(self.xNdot(t, args, **kwargs), t, args, x0, y0)
    
    def field_point(self, value, t, args=(), x0=0, y0=0):
        f = lambda q, *args: self.bohm_field.call(q, *args) - value
        res = sciopt.root(f, [x0, y0], jac = self.bohm_field.calljac, args=(t, *args))
        return res.x
    
    def eigen_lines(self, t, x0, y0, s, epsilon=1e-6, safety_dist=1e-5, curve_length=True, rich=False, **kwargs):
        kwargs['args'] = (t,) + kwargs.get('args', ())
        return self.bohm_field.eigen_lines(x0, y0, s, epsilon, safety_dist, curve_length, rich, **kwargs)

    def uv_field(self, t, args=(), **kwargs):
        to_sub = {self.args[i]: args[i] for i in range(len(self.args))}
        xN, yN = self.node(t, args, **kwargs)
        xNdot, yNdot = self.xNdot(t, args, **kwargs)
        u, v = variables('u, v')
        Udot = self.bohm_field.x.expr.subs(to_sub).subs({self.xvar: u+xN, self.yvar: v+yN, self.tvar: t}) - xNdot
        Vdot = self.bohm_field.y.expr.subs(to_sub).subs({self.xvar: u+xN, self.yvar: v+yN, self.tvar: t}) - yNdot
        return ConservativeVectorField2D([Udot, Vdot], u, v)
    
    @cached_property
    def _xpoint_sys(self):
        vx, vy = Dummy('vx'), Dummy('vy')
        return EquationSystem([self.bohm_field.x.expr - vx, self.bohm_field.y.expr - vy], [self.xvar, self.yvar], (vx, vy, self.tvar, *self.args), self._xpoint_bin, self._dir)
    
    def npxpc(self, t_span, rn0, rx0, xtol=1e-13, ftol=1e-13, max_iter=100, max_frames=-1, **odeargs):
        t0, t = t_span
        args = odeargs.get('args', ())
        xn, yn = sciopt.root(self._psi_eqsystem, rn0, args=(t0, *args), jac=self.jacPsi, options=dict(xtol=xtol)).x
        xn_orb = self.co_orbit(t0, xn, yn, **odeargs).go_to(t, max_frames=max_frames)
        rx_array = np.zeros_like(xn_orb.q)

        # try and get initial position of X point
        eq_sys = self._xpoint_sys
        xn_dot = self.v_co(t0, xn, yn, *args)
        res = eq_sys.newton_raphson(rx0, args=(*xn_dot, t0, *args), ftol=ftol, xtol=xtol, max_iter = max_iter)
        if not res.success:
            raise ValueError(f"npxpc failed to detect initial X point: Iters = {int(res.iters)}, x = {res.root}")        
        # initial X point found
        rx_array[0] = res.root
        for i, (ti, qi) in enumerate(zip(xn_orb.t[1:], xn_orb.q[1:])):
            xn_dot = self.v_co(ti, *qi, *args)
            res = eq_sys.newton_raphson(rx_array[i, :], args=(*xn_dot, ti, *args), ftol=ftol, xtol=xtol, max_iter = max_iter)
            if res.success:
                rx_array[i+1] = res.root
            else:
                return xn_orb.t[:i], xn_orb.q[:i], rx_array[:i]
        return xn_orb.t, xn_orb.q, rx_array
    
    @property
    def force_field(self):
        f = self._V + self.quantum_potential
        return ConservativeVectorField2D(f, self.xvar, self.yvar, self.tvar, *self.args)
    
    @cached_property
    def ode_system(self):
        return OdeSystem(self._odesys_data, self.tvar, [self.xvar, self.yvar], args=self.args, module_name=self._odesys_modname, directory=self._dir)
    
    @cached_property
    def nodal_point_odesys(self):
        return OdeSystem(self._v_co, self.tvar, [self.xvar, self.yvar], self.args, module_name=self._nodalpoint_modname, directory=self._dir)

    @cached_property
    def _v_co(self):
        a = Real(self.psi).diff(self.xvar)
        b = Real(self.psi).diff(self.yvar)
        c = Imag(self.psi).diff(self.xvar)
        d = Imag(self.psi).diff(self.yvar)
        D = a*d - b*c
        C_inv = np.array([[d, -b], [-c, a]], dtype=object)
        vec = np.array([Real(self.psi).diff(self.tvar), Imag(self.psi).diff(self.tvar)], dtype=object)
        rhs = -C_inv.dot(vec)/D
        return rhs
    
    @cached_property
    def _v_co_lambdaexpr(self):
        return VectorLambdaExpr(self._v_co, self.tvar, self.xvar, self.yvar, *self.args)
    
    def v_co(self, t, x, y, *args)->np.ndarray:
        return self._v_co_lambdaexpr(t, x, y, *args)
    
    def co_orbit(self, t0, x0, y0, **odeargs):
        return self.nodal_point_odesys.get(t0, [x0, y0], **odeargs)
    
    def varode_sys(self, DELTA_T):
        return VariationalBohmianSystem((self.psi, *self._varodesys_data), self.tvar, self.xvar, self.yvar, self.delx, self.dely, self.args, DELTA_T, module_name=self._varodesys_modname, directory=self._dir)
    
    def orbit(self, x0, y0, t0=0., rtol=0, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="DOP853"):
        s = self.ode_system
        return BohmianOrbit(LowLevelODE(s.lowlevel_odefunc, jac=s.lowlevel_jac, t0=t0, q0=np.array([x0, y0]), rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method))
    
    def variational_orbit(self, x0, y0, t0=0., rtol=0, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), DELTA_T=0.05, method='DOP853'):
        return self.varode_sys(DELTA_T=DELTA_T).get_orbit(x0, y0, t0=t0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method)
    
