from typing import Callable, TypeAlias
from abc import ABCMeta, abstractmethod

from pathos.multiprocessing import ProcessPool as Pool
import dill

import numpy as np
import numpy.typing as npt
from tqdm import tqdm


SDEFn: TypeAlias = Callable[[float], float]

class SDESolver(metaclass=ABCMeta):
    """
    Abstract base class for a solver that simulates SDEs.

    Attributes:
        a (SDEFn): The drift coefficient function.
        b (SDEFn): The diffusion coefficient function.
        N (int): Number of time steps.
        num_chains (int): Number of independent chains to simulate.
        dt (float): Time step size.
        num_workers (int): Number of worker threads to use for parallel execution.
        dW (Callable): Function to generate random increments for the Wiener process.

    Methods:
        step(dt: float, dW: SDEFn, Y_prev: float) -> npt.NDArray[np.float64]:
            Abstract method to perform a single step of the SDE solver.
        
        run() -> npt.NDArray[np.float64]:
            Runs the solver for the specified number of chains and time steps, returning the results as a NumPy array.
    """

    def __init__(self, a: SDEFn, b: SDEFn, t_start: int, t_stop: int, num_chains: int, num_workers: int = 1, Y0: float = 0.0):
        """
        Initializes the SolverBase with the given parameters.

        Args:
            a (SDEFn): The drift coefficient function.
            b (SDEFn): The diffusion coefficient function.
            t_start (int): Start time of the simulation.
            t_stop (int): Stop time of the simulation.
            num_chains (int): Number of independent chains to simulate.
            num_workers (int): Number of worker threads to use for parallel execution (default is 1).
            Y0 (float): starting point for chain.
        """

        self.a = a
        self.b = b
        self.N = (t_stop-t_start)+1
        self.num_chains = num_chains
        self.Y0 = Y0
        self.dt = 1/self.N
        self.num_workers = num_workers
        self.dW = lambda _ : np.random.normal(loc=0.0, scale=np.sqrt(self.dt))

        # Serializer settings
        dill.settings['recurse'] = True

    @abstractmethod
    def step(self, Y_prev: float) -> npt.NDArray[np.float64]:
        """
        Method to perform a single step of the SDE solver.

        Args:
            Y_prev (float): Previous value of the process.

        Returns:
            npt.NDArray[np.float64]: The new value of the process after the step.
        """

        pass

    def run(self) -> npt.NDArray[np.float64]:
        """
        Runs the solver for the specified number of chains and time steps.

        Returns:
            npt.NDArray[np.float64]: A NumPy array containing the results of the simulation for each chain on axis=0.
        """
        
        def chain(i: int) -> npt.NDArray[np.float64]:
            # For multiprocessing
            rng = np.random.default_rng(seed=i)
            self.dW = lambda _ : rng.normal(loc=0.0, scale=np.sqrt(self.dt))

            N = self.N
            Y = np.zeros(N)
            Y[0] = self.Y0
            for i in tqdm(range(1, N), desc=f"Chain {i}"):
                Y[i] = self.step(Y[i-1])
            return Y
        
        chains = range(1, self.num_chains+1)
        if self.num_workers > 1:

            with Pool(self.num_workers) as pool:
                Ys = pool.map(chain, chains)
        else:
            Ys = [chain(i) for i in chains]
        return np.stack(Ys)
