import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from .ir_model import IRModel
from solver import EulerMaruyama

class BlackKarasinski(IRModel):
    """
    Black-Karasinski interest rate model.
    
    Attributes:
        theta (list[float]): Mean reversion level parameters.
        phi (list[float]): Mean reversion speed parameters.
        sigma (list[float]): Volatility parameters.
        r0 (float): Initial interest rate.

    Methods:
        __init__(self, theta: list[float], phi: list[float], sigma: list[float], r0: float):
            Initializes the Black-Karasinski model with given parameters.
        a(self, Y_prev: float, t: float) -> float:
            Drift term of the model.
        Y0(self) -> float:
            Initial value of the interest rate.
        differentiable(self) -> bool:
            Indicates if the model is differentiable.
        b(self, Y_prev: float, t: float) -> float:
            Diffusion term of the model.
        calibrate(self, rates: npt.NDArray[np.float64], maxiter: int = 10):
            Calibrates the model parameters to fit the given interest rate data.
    """
    

    def __init__(self, theta: list[float], phi: list[float], sigma: list[float], r0: float):
        super().__init__()
        self.theta = theta
        self.phi = phi
        self.sigma = sigma
        self.r0 = r0

    def a(self, Y_prev: float, t: float) -> float:
        return self.theta[int(t*len(self.theta))]-self.phi[int(t*len(self.phi))]*Y_prev

    def Y0(self) -> float:
        return self.r0

    def differentiable(self) -> bool:
        return False

    def b(self, Y_prev: float, t: float) -> float:
        return self.sigma[int(t*len(self.sigma))]

    def calibrate(self, rates: npt.NDArray[np.float64], maxiter: int = 10):
        N = len(rates)
        dt = 1/N

        self.r0 = rates[0]

        def loss(x, r0, rates, N):
            theta, phi, sigma = x[:N], x[N:2*N], x[2*N:]
            bk = BlackKarasinski(theta, phi, sigma, r0)
            solver = EulerMaruyama(bk.a, bk.b, 0, N-1, 1, 1, r0)
            Y = solver.run()
            return np.sum((Y - rates)**2)
        
        theta0 = np.ones(N) * 0.05
        phi0 = np.ones(N) * 0.3
        sigma0 = np.ones(N) * 0.1
        x0 = np.concatenate([theta0, phi0, sigma0])

        res = minimize(loss, x0, args=(self.r0, rates, N), options={'maxiter':maxiter})
        self.theta, self.phi, self.sigma = res.x[:N], res.x[N:2*N], res.x[2*N:]
 