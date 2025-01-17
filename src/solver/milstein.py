import numpy as np
import numpy.typing as npt
import sympy as sym

from .sde_solver import SDESolver, SDEFn


class Milstein(SDESolver):
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
        Y0 (float): starting point for chain.

    Methods:
        step(dt: float, dW: SDEFn, Y_prev: float) -> npt.NDArray[np.float64]:
            Perform a single Milstein step.
    """

    def __init__(self, a: SDEFn, b: SDEFn, t_start: int, t_stop: int, num_chains: int, num_workers: int = 1, Y0: float = 0.0):
        super().__init__(a, b, t_start, t_stop, num_chains, num_workers, Y0)
        x, y = sym.symbols('x, y')
        self.b_prime = sym.lambdify((x, y), sym.diff(self.b(x, y), x), "numpy")

    def step(self, Y_prev: float, t: float) -> npt.NDArray[np.float64]:
        dt = self.dt
        dW_val = self.dW(dt)
        b_val = self.b(Y_prev, t)
        return Y_prev + self.a(Y_prev, t)*dt + b_val*dW_val+ 0.5*b_val*self.b_prime(Y_prev, t)*((dW_val**2)-dt)
