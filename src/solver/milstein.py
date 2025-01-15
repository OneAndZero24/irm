import numpy as np
import numpy.typing as npt

from .solver_base import SolverBase, SDEFn


class Milstein(SolverBase):
    """
    Milstein method for solving Stochastic Differential Equations (SDEs).

    Attributes:
        a (SDEFn): Drift coefficient function.
        b (SDEFn): Diffusion coefficient function.
        b_prime (SDEFn): Derivative of the diffusion coefficient function.
        t_start (int): Start time.
        t_stop (int): Stop time.
        N (int): Number of time steps.
        num_chains (int): Number of chains.
        num_workers (int): Number of workers for parallel computation.

    Methods:
        step(dt: float, dW: SDEFn, Y_prev: float) -> npt.NDArray[np.float64]:
            Perform a single Milstein step.
    """

    def __init__(self, a: SDEFn, b: SDEFn, b_prime: SDEFn, t_start: int, t_stop: int, N: int, num_chains: int, num_workers: int = 1):
        super().__init__(t_start, t_stop, N, num_chains, num_workers)
        self.a = a
        self.b = b
        self.b_prime = b_prime

    def step(self, Y_prev: float) -> npt.NDArray[np.float64]:
        dt = self.dt
        dW_val = self.dW(dt)
        b_val = self.b(Y_prev)
        return Y_prev + self.a(Y_prev)*dt + b_val*dW_val+ 0.5*b_val*self.b_prime(Y_prev)*((dW_val**2)-dt)
