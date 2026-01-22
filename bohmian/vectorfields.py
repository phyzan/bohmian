from __future__ import annotations
import warnings
import numpy as np
import scipy.optimize as sciopt
import scipy.integrate as scint
from numiphy.findiffs import grids
from numiphy.symlib import *
from numiphy.toolkit import *
from odepack import *


class VectorField2D:


    def __init__(self, Fx: Expr, Fy: Expr, x: Symbol, y: Symbol, *args: Symbol):
        symbols = []
        for xi in (x, y)+args:
            if xi in symbols:
                raise ValueError('Repeated symbols')
            symbols.append(xi)
            
        self.xvar, self.yvar = x, y
        self.x = ScalarLambdaExpr(Fx, *symbols)
        self.y = ScalarLambdaExpr(Fy, *symbols)

        self.Jac = VectorLambdaExpr([[Fx.diff(x), Fx.diff(y)], [Fy.diff(x), Fy.diff(y)]], *symbols)
        self.args = args


    def __call__(self, *args, **kwargs):
        return np.array([self.x(*args, **kwargs), self.y(*args, **kwargs)])

    def __mul__(self, other: float):
        return VectorField2D(other*self.x.expr, other*self.y.expr, self.xvar, self.yvar)
    
    def __rmul__(self, other: float):
        return self*other
    
    def unitvec(self, *args, **kwargs):
        vec = self(*args, **kwargs)
        return vec/np.sqrt(vec[0]**2+vec[1]**2)

    def call(self, q, *args):
        return self(*q, *args)
    
    def calljac(self, q, *args):
        return self.Jac(*q, *args)
    
    def fixed_point(self, args=(), x0=0, y0=0):
        return sciopt.root(self.call, [x0, y0], jac = self.calljac, args=args).x

    def flowdot(self, line: Line2D):
        return lambda u, *args: self.x(line.x(u), line.y(u), *args)*line.xdot(u) + self.y(line.x(u), line.y(u), *args)*line.ydot(u)

    def flow(self, line: Line2D, *args)->float|complex:
        cdot = self.flowdot(line)
        return scint.quad(cdot, *line.lims, args=args, epsabs=1e-10)[0]
    
    def streamline(self, x0, y0, s, curve_length=True, rich=False, compiled=True, **kwargs):
        '''
        Let F be a vector field.
        A field line R(s) passing through a point (x0, y0) satisfies the equation

        dR/ds = F(R), with initial conditions R(s) = (x0, y0)

        The "s" parameter is the more useful curve-length parameter, if we instead choose the ode:
        dR/ds = F(R) / |F(R)|
        which means dR/ds is the unit vector of the vector field at each point
        '''
        t = Symbol("_t_int")
        fx, fy = self.x.expr, self.y.expr
        if curve_length:
            A = (fx**2 + fy**2)**Rational(1, 2)
        else:
            A = 1
        s, direction = abs(s), 1 if s>0 else (-1 if s < 0 else 0)
        if direction > 0:
            kwargs['direction'] = kwargs.get('direction', 1)
        else:
            kwargs['direction'] = direction
        odesys = OdeSystem([fx/A, fy/A], t, [self.xvar, self.yvar], self.args, events=kwargs.pop('events', ()))
        ode, kwargs = call_with_consumed(odesys.get, t0=0, q0=[x0, y0], compiled=compiled, **kwargs)
        if rich:
            res, kwargs = call_builtin_with_consumed(ode.rich_integrate, dict(event_options=[], max_prints=0), s, **kwargs)
        else:
            res, kwargs = call_builtin_with_consumed(ode.integrate, dict(t_eval=None, event_options=[], max_prints=0), s, **kwargs)
        if kwargs:
            warnings.warn(f"The keyword arguments {kwargs} had no effect")
        return res
        
    def loop(self, q, r, *args):
        c = Circle(r, q)
        return self.flow(c, *args)
    
    def plot(self, grid: grids.Grid, args=(), scaled=True, **kwargs):

        def partial_update(xlims, ylims):
            if xlims[0] is None or xlims[1] is None:
                xlims = ax.get_xlim()
            if ylims[0] is None or ylims[1] is None:
                ylims = ax.get_ylim()
            ax.clear()

            qp: QuiverPlot = figure.artists[0]
            qp.args = get_quiver_data(xlims, ylims)
            figure.draw(ax)

            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            fig.set_size_inches(figure.figsize)

        def update(event):
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            partial_update(xlims, ylims)

        def get_quiver_data(xlims, ylims):
            x = np.linspace(*xlims, grid.shape[0])
            y = np.linspace(*ylims, grid.shape[1])
            xmesh = np.meshgrid(x, y, indexing='xy')
            X, Y = self(*xmesh, *args)
            mag = np.sqrt(X**2 + Y**2)
            with np.errstate(invalid='ignore'):
                X = np.where(X!=0, X/mag, 0)
                Y = np.where(Y!=0, Y/mag, 0)
            res = x, y, X, Y
            if scaled:
                res += (mag,)
            return res

        figure = SquareFigure("VectorField", figsize=(5, 5))
        figure.ftype = 'pdf'
        figure.xlabel = f'${self.xvar}$'
        figure.ylabel = f'${self.yvar}$'
        figure.yrot = 0
        figure.fontsize['label'] = 15
        figure.grid = dict(visible=False)
        figure.xlims = grid.limits[0]
        figure.ylims = grid.limits[1]

        figure.aspect = 'equal'
        figure.add(QuiverPlot(*get_quiver_data(*grid.limits), **kwargs))

        fig, ax = figure.plot()
        if figure.xlims[0] is not None and figure.xlims[1] is not None:
            ax.set_xlim(*figure.xlims)
        if figure.ylims[0] is not None and figure.ylims[1] is not None:
            ax.set_ylim(*figure.ylims)
        fig.canvas.mpl_connect('button_release_event', update)
        return figure, fig, ax, lambda : partial_update(figure.xlims, figure.ylims)
    


class ConservativeVectorField2D(VectorField2D):

    def __init__(self, arg1: Expr | tuple[Expr, Expr], x, y, *args):
        '''
        arg1: V such that F = grad V, or Fx, Fy
            If arg1 = (Fx, Fy), it is the user's responsibility that f is indeed conservative
        '''
        if isinstance(arg1, Expr):
            Fx, Fy = arg1.diff(x), arg1.diff(y)
        else:
            Fx, Fy = arg1
        VectorField2D.__init__(self, Fx, Fy, x, y, *args)

    def eigen_lines(self, x0, y0, s, epsilon = 1e-6, safety_dist = 1e-5, curve_length=True, rich=False, compiled=True, **kwargs):
        args = kwargs.get('args', ())
        kwargs_0 = kwargs.copy()
        xn, yn = self.fixed_point(args, x0, y0)
        v_n = np.array([xn, yn])
        eigres = np.linalg.eigh(self.Jac(xn, yn, *args))
        l1, l2 = eigres.eigenvalues
        v1, v2 = eigres.eigenvectors.T
        if abs(l1) < 1e-10 or abs(l2) < 1e-10:
            raise ValueError('Zero eigenvalue in Jacobian. Cannot linearize.')
        x, y = self.xvar, self.yvar
        if rich:
            res: dict[float, tuple[OdeSolution, OdeSolution]] = {l1: (), l2: ()}
        else:
            res: dict[float, tuple[OdeResult, OdeResult]] = {l1: (), l2: ()}
        for l, v in zip([l1, l2], [v1, v2]):
            forwards_integration = l > 0
            direction = 1 if forwards_integration else -1
            event = approach_point_event("_loop", x, y, xn, yn, safety_dist, forwards_integration)
            opt = EventOpt("_loop", max_events=1, terminate=True)
            kwargs['events'] = tuple(kwargs_0.get('events', ())) + (event,)
            kwargs['event_options'] = tuple(kwargs_0.get('event_options', ())) + (opt,)
            for sgn in [1, -1]:
                v0 = v_n + sgn*epsilon * v
                res_tmp = self.streamline(v0[0], v0[1], s, curve_length=curve_length, rich=rich, direction=direction, compiled=compiled, **kwargs)
                res[l] += (res_tmp,)
        return res


def approach_point_event(name, x, y, x_point, y_point, dr, forwards_integration=True):
    direction = -1 if forwards_integration else 1
    return SymbolicPreciseEvent(name, (x-x_point)**2 + (y-y_point)**2 - dr**2, direction)