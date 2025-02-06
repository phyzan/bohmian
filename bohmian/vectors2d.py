from __future__ import annotations
# from .. import symbolic as sym
from ..numiphy.findiffs import grids
from ..numiphy.odesolvers import ode_solvers as ods
import numpy as np
import math
from typing import Callable
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
import scipy.integrate as scint
from ..numiphy import expressions as sym



class Line2D:

    def __init__(self, x: sym.Expr, y: sym.Expr, u: sym.Variable, a: float, b: float):
        self._x = sym.lambdify('numpy', x, [u])[0]
        self._y = sym.lambdify('numpy', y, [u])[0]

        self._xdot = sym.lambdify('numpy', x.diff(u), [u])[0]
        self._ydot = sym.lambdify('numpy', y.diff(u), [u])[0]
        self.a, self.b = a, b

    def x(self, u):
        return self._x(u)
    
    def y(self, u):
        return self._y(u)
    
    def xdot(self, u):
        return self._xdot(u)
    
    def ydot(self, u):
        return self._ydot(u)


class Parallelogram(Line2D):

    def __init__(self, x1, x2, y1, y2):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.a, self.b = 0, 4
        self.Lx = x2-x1
        self.Ly = y2-y1
        self.center = [(x1+x2)/2, (y1+y2)/2]

    def x(self, u):
        x0, a = self.x1, self.x2 - self.x1
        if 0 <= u < 1:
            return x0 + u*a
        elif 1 <= u < 2:
            return x0 + a
        elif 2 <= u < 3:
            return x0 + a*(3-u)
        else:
            return x0
        
    def y(self, u):
        y0, a = self.y1, self.y2 - self.y1
        if 0 <= u < 1:
            return y0
        elif 1 <= u < 2:
            return y0 + (u-1)*a
        elif 2 <= u < 3:
            return y0 + a
        else:
            return y0 + a*(4-u)
        
    def xdot(self, u):
        a = self.x2 - self.x1
        if 0 <= u < 1:
            return a
        elif 1 <= u < 2:
            return 0
        elif 2 <= u < 3:
            return -a
        else:
            return 0

    def ydot(self, u):
        a = self.y2 - self.y1
        if 0 <= u < 1:
            return 0
        elif 1 <= u < 2:
            return a
        elif 2 <= u < 3:
            return 0
        else:
            return -a
        
    def split(self)->list[Parallelogram]:
        a, b, c, d = self.x1, self.x2, self.y1, self.y2
        shapes = []
        xaxis = [(a, (a+b)/2), ((a+b)/2, b)]
        yaxis = [(c, (c+d)/2), ((c+d)/2, d)]
        for x in xaxis:
            for y in yaxis:
                shapes.append(Parallelogram(*x, *y))
        return shapes


class VectorField2D:


    def __init__(self, Fx: sym.Expr, Fy: sym.Expr, x:sym.Variable, y:sym.Variable, property=None):
        self.xvar, self.yvar = x, y
        v = sum(Fx.variables+Fy.variables).reordered_vars(x, y)
        self.x = sym.ScalarLambdaExpr('numpy', Fx, v, property=property)
        self.y = sym.ScalarLambdaExpr('numpy', Fy, v, property=property)

        div, curl = Fx.diff(x) + Fy.diff(y), Fy.diff(x) - Fx.diff(y)
        self.div = sym.ScalarLambdaExpr('numpy', div, v, property=property)
        self.curl = sym.ScalarLambdaExpr('numpy', curl, v, property=property)

        self.Jac = sym.VectorLambdaExpr('numpy', [[Fx.diff(x), Fx.diff(y)], [Fy.diff(x), Fy.diff(y)]], v, property=property)
        self.property = property


    def __call__(self, *args, **kwargs):
        return np.array([self.x(*args, **kwargs), self.y(*args, **kwargs)])

    def __mul__(self, other: float):
        return VectorField2D(other*self.x.expr, other*self.y.expr, self.xvar, self.yvar, self.property)
    
    def __rmul__(self, other: float):
        return self*other
    
    def unitvec(self, *args, **kwargs):
        vec = self(*args, **kwargs)
        return vec/np.sqrt(vec[0]**2+vec[1]**2)

    def call(self, q):
        return self(*q)
    
    def calljac(self, q):
        return self.Jac(*q)
    
    def fixed_point(self, x0, y0, *args):
        return sciopt.root(self.call, [x0, y0], jac = self.calljac, args=args).x

    def flowdot(self, line: Line2D):
        return lambda u, *args: self.x(line.x(u), line.y(u), *args)*line.xdot(u) + self.y(line.x(u), line.y(u), *args)*line.ydot(u)

    def flow(self, line: Line2D, *args)->float|complex:
        cdot = self.flowdot(line)
        return scint.quad(cdot, line.a, line.b, args=args, epsabs=1e-10)[0]
    
    def streamline(self, x0, y0, s, ds=1e-3, err=1e-8, *args):
        '''
        Let F be a vector field.
        A field line R(s) passing through a point (x0, y0) satisfies the equation

        dR/ds = F(R), with initial conditions R(s) = (x0, y0)

        The "s" parameter is the more useful curve-length parameter, if we instead choose the ode:
        dR/ds = F(R) / |F(R)|
        which means dR/ds is the unit vector of the vector field at each point
        '''
        res = ods.ODE(lambda s, q: self.unitvec(*q, *args), ics=(0, np.array([x0, y0]))).solve(s, ds, err=err, display=False)
        return res[1].transpose()
        
    def loop(self, q, r, *args):
        c = Circle2D(*q, r)
        return self.flow(c, *args)
    
    def plot(self, grid: grids.Grid, *args, **kwargs):

        def update(event):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            lines = ax.get_lines()
            ax.clear()
            x = np.linspace(*xlim, grid.shape[0])
            y = np.linspace(*ylim, grid.shape[1])
            xmesh = np.meshgrid(x, y, indexing='ij')
            X, Y = self(*xmesh, *args)
            mag = np.sqrt(X**2+Y**2)
            ax.quiver(*xmesh, X/mag, Y/mag, mag, **kwargs)
            for line in lines:
                ax.add_line(line)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel(str(self.xvar), fontsize=20)
        ax.set_ylabel(str(self.yvar), fontsize=20)

        xmesh = grid.x_mesh()
        X, Y = self(*xmesh, *args)
        mag = np.sqrt(X**2+Y**2)

        ax.quiver(*xmesh, X/mag, Y/mag, mag, **kwargs)
        fig.canvas.mpl_connect('button_release_event', update)
        return fig, ax
    
    def plot_line(self, line: Line2D, grid: grids.Grid, n=400, *args, **kwargs):
        fig, ax = VectorField2D.plot(self, grid, *args, **kwargs)
        u = np.linspace(line.a, line.b, n)
        ax.plot(line.x(u), line.y(u))
        return fig, ax
    
    def plot_circle(self, q, r, grid: grids.Grid, n=400, *args, **kwargs):
        c = Circle2D(*q, r)
        return VectorField2D.plot_line(self, c, grid, n, *args, **kwargs)
