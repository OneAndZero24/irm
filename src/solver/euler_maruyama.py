import numpy as np
import numpy.typing as npt

from .solver_abc import SolverBase, SDEFn


class EulerMaruyama(SolverBase):
    """
    Euler-Maruyama method for solving stochastic differential equations (SDEs).
    This class implements the Euler-Maruyama method, which is a numerical method
    for solving SDEs of the form:
        dY = a(Y) dt + b(Y) dW

    Attributes:
        a (SDEFn): The drift coefficient function.
        b (SDEFn): The diffusion coefficient function.
        t_start (int): The start time of the simulation.
        t_stop (int): The stop time of the simulation.
        N (int): The number of time steps.
        num_chains (int): The number of independent chains to simulate.
        num_workers (int): The number of parallel workers to use for simulation.
        
    Methods:
        step(Y_prev: float) -> npt.NDArray[np.float64]:
            Perform a single Euler-Maruyama step.
    """

    def __init__(self, a: SDEFn, b: SDEFn, t_start: int, t_stop: int, N: int, num_chains: int, num_workers: int = 1):
        super().__init__(t_start, t_stop, N, num_chains, num_workers)
        self.a = a
        self.b = b

    def step(self, Y_prev: float) -> npt.NDArray[np.float64]:
        dt = self.dt
        return Y_prev + self.a(Y_prev)*dt + self.b(Y_prev)*self.dW(dt)
    