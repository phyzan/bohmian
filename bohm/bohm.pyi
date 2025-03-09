from numiphy.odesolvers.odepack import *
import numpy as np

class BohmianOrbit(LowLevelODE):

    def __init__(self, t0, q0, args, stepsize, *, rtol=1e-6, atol=1e-12, min_step=0., checkpoints=()):...

    @property
    def events(self)->np.ndarray[int]:...