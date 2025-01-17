import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from sympy import sqrt

from .ir_model import IRModel


class CIR(IRModel):
    """
    CIR Model for Interest Rate Modeling
    This class implements the Cox-Ingersoll-Ross (CIR) model, which is used for modeling interest rates.

    Attributes:
        theta (float): Long-term mean level.
        alpha (float): Speed of reversion to the mean.
        sigma (float): Volatility parameter.
        r0 (float): Initial interest rate.

    Methods:
        __init__(theta: float, alpha: float, sigma: float, r0: float):
            Initializes the CIR model with the given parameters.

        a(Y_prev: float, t: float) -> float:
            Computes the drift term of the CIR model.

        Y0() -> float:
            Returns the initial interest rate.

        b(Y_prev: float, t: float) -> float:
            Computes the diffusion term of the CIR model.
            
        calibrate(rates: npt.NDArray[np.float64]):
            Calibrates the CIR model parameters using historical interest rate data.
    """

    def __init__(self, theta: float, alpha: float, sigma: float, r0: float):
        super().__init__()
        self.theta = theta
        self.alpha = alpha
        self.sigma = sigma
        self.r0 = r0
     
    def a(self, Y_prev: float, t: float) -> float:
        """
        Computes the drift term of the CIR model.
        """

        return self.theta-self.alpha*Y_prev

    def Y0(self) -> float:
        """
        Returns the initial interest rate.
        """

        return self.r0

    def b(self, Y_prev: float, t: float) -> float:
        """
        Computes the diffusion term of the CIR model.
        """

        return self.sigma*sqrt(Y_prev)

    def calibrate(self, rates: npt.NDArray[np.float64]):
        """
        Calibrates the CIR model parameters using historical interest rate data.
        """

        N = len(rates)
        dt = 1/N

        rs = rates[:N - 1]  
        rt = rates[1:N]
        
        model = LinearRegression()

        y = (rt - rs) / np.sqrt(rs)
        z1 = dt / np.sqrt(rs)
        z2 = dt * np.sqrt(rs)
        X = np.column_stack((z1, z2))

        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        y_hat = model.predict(X)
        residuals = y - y_hat
        beta1 = model.coef_[0]        
        beta2 = model.coef_[1]

        k0 = -beta2
        theta0 = beta1/k0
        sigma0 = np.std(residuals)/np.sqrt(dt)

        self.theta = k0*theta0
        self.alpha = k0
        self.sigma = sigma0
        self.r0 = rates[0]
 
