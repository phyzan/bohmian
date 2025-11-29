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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numiphy.symlib.pylambda import ScalarLambdaExpr, VectorLambdaExpr
from .orbits import *
from rootfinder import *
from mcpy import PDF2D
from typing import Literal


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
            self._nodal_point_algebraic_system_modname = 'nodal_point_alg_sys'
            self._dir = directory
        else:
            self._odesys_modname = None
            self._varodesys_modname = None
            self._nodalpoint_modname = None
            self._xpoint_bin = None
            self._nodal_point_algebraic_system_modname = None
            self._dir = directory = None

        self._cached_var_ode_sys: dict[float, VariationalBohmianSystem] = {}
    
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
    
    def draw_ics(self, N: int, t: float, xlims: tuple[float, float], ylims: tuple[float, float], args: tuple[float, ...], distribution='Born'):
        if distribution=='Born':
            rho = self.rho.lambdify(self.xvar, self.yvar, self.tvar, *self.args, lib='math')
            rho_xy = lambda x, y: rho(x, y, t, *args)
            pdf = PDF2D(rho_xy, [xlims, ylims])
            return pdf.draw(N, therm_factor=10)
        elif distribution == 'uniform':
            return np.random.uniform(xlims, ylims, (N, 2))
        else:
            raise NotImplementedError
    
    def draw_orbits(self, N: int, t: float, xlims: tuple[float, float], ylims: tuple[float, float], distribution: Literal['Born', 'uniform'] = 'Born', args = (), dtype='double', **odeargs)->list[BohmianOrbit]:
        ics =self.draw_ics(N=N, t=t, xlims=xlims, ylims=ylims, args=args, distribution=distribution)
        orbs = [self.orbit(*ic, t0=t, args=args, dtype=dtype, **odeargs) for ic in ics]
        return orbs
    
    def draw_variational_orbits(self, N: int, t: float, xlims: tuple[float, float], ylims: tuple[float, float], distribution: Literal['Born', 'uniform'] = 'Born', args=(), DELTA_T=0.05, dtype='double', **odeargs)->list[VariationalBohmianOrbit]:
        ics = self.draw_ics(N=N, t=t, xlims=xlims, ylims=ylims, args=args, distribution=distribution)
        model = self.varode_sys(DELTA_T=DELTA_T)
        return [model.get_orbit(*ic, t0=t, args=args, dtype=dtype, **odeargs) for ic in ics]
    
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

    @cached_property
    def node_finder(self):
        return EquationSystem([Real(self.psi), Imag(self.psi)], [self.xvar, self.yvar], (self.tvar, *self.args), module_name=self._nodal_point_algebraic_system_modname, directory=self._dir)

    def node(self, t, args, Nsub = 100, rmax=1e-2, box = [-10, 10, -10, 10])->np.ndarray:
        sq = Parallelogram(*box)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=IntegrationWarning)
            flow = self.flow(sq, t, *args)

        error = lambda flow: ArithmeticError(f'No indication of vortex within the search limits of the Vector Field at t = {t}. Flow is {flow}')
        if abs(flow)>1e-3:
            for _ in range(Nsub):
                sqrs = sq.split()
                for i in range(4):
                    sq = sqrs[i]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=IntegrationWarning)
                        flow = self.flow(sq, t, *args)
                    if abs(flow) > 1e-4:
                        ri = 0.5*(sq.Lx**2 + sq.Ly**2)**0.5
                        if ri <= rmax:
                            res = self.node_finder.newton_raphson(sq.center, (t, *args))
                            if not res.success:
                                raise ValueError('Node search failed')
                            return res.root
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
    def field_point_finder(self):
        vx, vy = Dummy('vx'), Dummy('vy')
        return EquationSystem([self.bohm_field.x.expr - vx, self.bohm_field.y.expr - vy], [self.xvar, self.yvar], (vx, vy, self.tvar, *self.args), self._xpoint_bin, self._dir)
    
    def npxpc(self, t_span, rn0, rx0, xtol=1e-13, ftol=1e-13, max_iter=100, t_eval=None, **odeargs):
        t0, t = t_span
        args = odeargs.get('args', ())
        res_node_0 = self.node_finder.newton_raphson(rn0, (t0, *args), ftol=1e-14, xtol=1e-14, max_iter=100000)
        if not res_node_0.success:
            raise ValueError('Initial node location search failed in npxpc')
        xn, yn = res_node_0.root
        xn_orb = self.co_orbit(t0, xn, yn, **odeargs).go_to(t, t_eval=t_eval)
        rx_array = np.zeros_like(xn_orb.q)

        # try and get initial position of X point
        eq_sys = self.field_point_finder
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
    
    def orbit(self, x0, y0, t0=0., rtol=0, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), method="RK45", dtype='double'):
        s = self.ode_system
        return BohmianOrbit(LowLevelODE(s.lowlevel_odefunc(dtype=dtype), jac=s.lowlevel_jac(dtype=dtype), t0=t0, q0=np.array([x0, y0]), rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method), dtype=dtype)
    
    def variational_orbit(self, x0, y0, t0=0., rtol=0, atol=1e-12, min_step=0., max_step=np.inf, first_step=0., args=(), DELTA_T=0.05, method='RK45', dtype='double'):
        if DELTA_T not in self._cached_var_ode_sys:
            self._cached_var_ode_sys[DELTA_T] = self.varode_sys(DELTA_T=DELTA_T)
        
        return self._cached_var_ode_sys[DELTA_T].get_orbit(x0, y0, t0=t0, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, first_step=first_step, args=args, method=method, dtype=dtype)
    
    def rho_time_averaged(
        self,
        spatial_grid: grids.Grid,
        t_span: tuple[float, float],
        nt: int,
        chunk_size=2,
        args=(),
        threads=-1,
        device='cpu'
    ):
        # ----------------------------
        # 1. Set threads
        # ----------------------------
        if threads == -1:
            threads = torch.get_num_threads()  # max available
        torch.set_num_threads(threads)

        # ----------------------------
        # 2. Set device
        # ----------------------------
        device = torch.device(device)  # e.g., 'cpu', 'cuda', 'mps' or ROCm 'cuda'
        
        # ----------------------------
        # 3. Create torch-based callable
        # ----------------------------
        f = self.rho.subs({arg: given_arg for arg, given_arg in zip(self.args, args)}).lambdify(self.tvar, self.xvar, self.yvar, lib='torch')

        # ----------------------------
        # 4. Time array and step size
        # ----------------------------
        t_arr = torch.linspace(t_span[0], t_span[1], nt + 1, dtype=torch.float64, device=device)
        dt = (t_span[1] - t_span[0]) / nt

        # ----------------------------
        # 5. Spatial mesh
        # ----------------------------
        x_axis_points, y_axis_points = spatial_grid.x
        xmesh, ymesh = torch.meshgrid(
            torch.tensor(x_axis_points, dtype=torch.float64, device=device),
            torch.tensor(y_axis_points, dtype=torch.float64, device=device),
            indexing='ij'
        )

        # ----------------------------
        # 6. Prepare total and carry-over
        # ----------------------------
        total = torch.zeros_like(xmesh, dtype=torch.float64, device=device)
        prev = None

        # ----------------------------
        # 7. Process in time chunks
        # ----------------------------
        for i in range(0, len(t_arr), chunk_size):
            t_chunk = t_arr[i:i + chunk_size].view(-1, 1, 1)  # shape (chunk_size,1,1)
            X = xmesh[None, :, :]
            Y = ymesh[None, :, :]

            # Evaluate function (already torch, on the correct device)

            vals = f(t_chunk, X, Y)

            # Prepend carry-over if exists
            if prev is not None:
                vals = torch.cat([prev[None, :, :], vals], dim=0)

            # Simpson's rule over complete triples
            n_triples = (vals.shape[0] - 1) // 2
            for j in range(n_triples):
                total += (dt / 3.0) * (vals[2*j] + 4*vals[2*j + 1] + vals[2*j + 2])

            # Keep last point for next chunk
            prev = vals[-1] if vals.shape[0] % 2 == 1 else vals[-2]

        # ----------------------------
        # 8. Normalize by total time span
        # ----------------------------
        total /= (t_span[1] - t_span[0])

        # ----------------------------
        # 9. Convert back to NumPy for DummyScalarField
        # ----------------------------
        return DummyScalarField(total.cpu().numpy(), spatial_grid, self.xvar, self.yvar)
    
    def orbit_colormap_data(self, x0, y0, t_span: tuple[float, float], nt: int, max_prints=0, dtype='double', **odeargs)->OdeResult:
        t0, t = t_span
        t_eval = np.linspace(t0, t, nt+1, endpoint=True)
        return self.orbit(x0=x0, y0=y0, t0=t0, dtype=dtype, **odeargs).go_to(t, t_eval=t_eval, max_prints=max_prints)
    
    def multi_orbit_colormap(self, N: int, t_span: tuple[float, float], nt: int, xlims: tuple[float, float], ylims: tuple[float, float], bins: tuple[int, int], distribution: Literal['Born', 'uniform'] = 'Born', chunks: int = 1, args = (), first_step=0., dtype='double', **odeargs):
        t0, tmax = t_span
        orbs = self.draw_orbits(N=N, t=t0, xlims=xlims, ylims=ylims, distribution=distribution, args=args, first_step=first_step, dtype=dtype, **odeargs)
        xbins = np.linspace(*xlims, bins[0], endpoint=True)
        ybins = np.linspace(*ylims, bins[1], endpoint=True)

        #initialize accumulator
        hist = np.zeros((len(xbins)-1, len(ybins)-1), dtype=np.float64)
        chunk_interval = (tmax-t0)/chunks

        def data_batches():
            for k in range(chunks):
                t_eval = np.linspace(chunk_interval*k, chunk_interval*(k+1), nt//chunks, endpoint=False)
                for i, orb in enumerate(orbs):
                    solver = orb.solver()
                    orbs[i] = self.orbit(*solver.q, t0=solver.t, first_step=solver.stepsize, args=args, dtype=dtype, **odeargs)
                integrate_all(orbs, interval=chunk_interval, t_eval=t_eval)
                xdata, ydata = [np.concatenate([orbi.q.T[0] for orbi in orbs]), np.concatenate([orbi.q.T[1] for orbi in orbs])]
                yield xdata, ydata
        
        for batch in data_batches():
            h, _, _ = np.histogram2d(*batch, bins=[xbins, ybins])
            hist += h

        return xbins, ybins, hist
    
    def plot_multi_orbit_colormap(self, N: int, t_span: tuple[float, float], nt: int, xlims: tuple[float, float], ylims: tuple[float, float], bins: tuple[int, int], distribution: Literal['Born', 'uniform'] = 'Born', chunks: int = 1, density=True, args = (), first_step=0., rtol=0, atol=1e-8, min_step=0., max_step=np.inf, method='RK45', dtype='double', **plotargs):
        xbins, ybins, hist = self.multi_orbit_colormap(N=N, t_span=t_span, nt=nt, xlims=xlims, ylims=ylims, bins=bins, distribution=distribution, chunks=chunks, args=args, first_step=first_step, rtol=rtol, atol=atol, min_step=min_step, max_step=max_step, method=method, dtype=dtype)

        if density:
            hist /= (hist.sum() * np.outer(np.diff(xbins), np.diff(ybins)))

        grid = grids.Uniform1D(*xlims, bins[0]) * grids.Uniform1D(*ylims, bins[1])
        fig, ax, cbar = plot(hist, grid, **plotargs)
        return fig, ax, cbar

def orbit_colormap(xdata, ydata, **hist_args):

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')

    # Create 2D histogram
    h, xedges, yedges, im = ax.hist2d(xdata, ydata,**hist_args)

    # Use make_axes_locatable to ensure perfect alignment
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add colorbar without exceeding the main axis bounds
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f'$\\tilde{{\\rho}}$', labelpad=2)
    fig.tight_layout(pad=1.2)
    
    return fig, ax, cbar