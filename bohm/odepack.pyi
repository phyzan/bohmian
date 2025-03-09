import numpy as np
from typing import Callable
import numpy as np


Func = Callable[[float, np.ndarray], np.ndarray]
ObjFunc = Callable[[float, np.ndarray], float]
BoolFunc = Callable[[float, np.ndarray], bool]


class Event:

    def __init__(self, name: str, when: ObjFunc, check_if: BoolFunc=None, mask: Func=None):...


class StopEvent:

    def __init__(self, name: str, when: ObjFunc, check_if: BoolFunc=None):...


class OdeResult:

    @property
    def t(self)->np.ndarray:...

    @property
    def q(self)->np.ndarray:...

    @property
    def events(self)->dict[str, np.ndarray]:...

    @property
    def diverges(self)->bool:...

    @property
    def is_stiff(self)->bool:...

    @property
    def success(self)->bool:...

    @property
    def runtime(self)->float:...

    @property
    def message(self)->str:...

    def examine(self):...


class SolverState:

    @property
    def t(self)->float:...

    @property
    def q(self)->np.ndarray:...

    @property
    def event_map(self)->bool:...

    @property
    def diverges(self)->bool:...

    @property
    def is_stiff(self)->bool:...

    @property
    def is_running(self)->bool:...

    @property
    def is_dead(self)->bool:...

    @property
    def N(self)->int:...

    @property
    def message(self)->str:...

    def show(self):...


class LowLevelODE:

    def __init__(self, f, t0, q0, stepsize, *, rtol=1e-6, atol=1e-12, min_step=0., args=(), method="RK45", event_tol=1e-12, events: list[Event]=None, stop_events: list[StopEvent] = None):...

    def integrate(self, interval, *, max_frames=-1, max_events=-1, terminate=True, display=False)->OdeResult:...

    def advance(self)->SolverState:...

    def state(self)->SolverState:...

    def copy(self)->LowLevelODE:...

    @property
    def t(self)->np.ndarray:...

    @property
    def q(self)->np.ndarray:...

    @property
    def event_map(self)->dict[str, np.ndarray]:...

    @property
    def runtime(self)->float:...

    @property
    def diverges(self)->bool:...

    @property
    def is_stiff(self)->float:...

    @property
    def is_dead(self)->float:...


def integrate_all(ode_array, interval, max_frames=-1, max_events=-1, terminate=True):...