from .ir_model import IRModel

import numpy as np
import numpy.typing as npt


class Vasicek(IRModel):
    """
    Vasicek interest rate model.
    The Vasicek model is a type of one-factor short rate model that describes the evolution of interest rates. 
    It is defined by the following stochastic differential equation:
        dY_t = (theta - alpha*Y_t) * dt + sigma * dW_t
    where:
    - Y_t is the short rate at time t
    - alpha is the speed of reversion
    - theta is the long-term mean level
    - sigma is the volatility
    - W_t is a Wiener process (standard Brownian motion)

    Attributes:
        theta (float): The long-term mean level of the interest rate.
        alpha (float): The speed of reversion to the mean.
        sigma (float): The volatility of the interest rate.
        r0 (float): First rate.

    Methods:
        a(Y_prev: float) -> float:
            Computes the drift term of the Vasicek model.

        b(Y_prev: float) -> float:
            Computes the diffusion term of the Vasicek model.

        Y0() -> float:
            Starting point for chain.

        calibrate(data: pd.DataFrame):
            Calibrates the model parameters to market data.
    """

    def __init__(self, theta: float, alpha: float, sigma: float, r0: float):
        """
        Initializes the Vasicek model with the given parameters.

        Args:
            theta (float): The long-term mean level of the interest rate.
            alpha (float): The speed of reversion to the mean.
            sigma (float): The volatility of the interest rate.
            r0 (float): First rate.
        """

        super().__init__()
        self.theta = theta
        self.alpha = alpha
        self.sigma = sigma
        self.r0 = r0
     
    def a(self, Y_prev: float) -> float:
        """
        Computes the drift term of the Vasicek model.
        Args:
            Y_prev (float): The previous value of the interest rate.
        Returns:
            float: The drift term.
        """

        return self.theta-self.alpha*Y_prev

    def b(self, Y_prev: float) -> float:
        """
        Computes the diffusion term of the Vasicek model.
        Args:
            Y_prev (float): The previous value of the interest rate.
        Returns:
            float: The diffusion term.
        """

        return self.sigma
    
    def Y0(self) -> float:
        """
        Starting point for chain.
        """
        
        return self.r0

    def calibrate(self, t_start: int, t_stop: int, rates: npt.NDArray[np.float64]):
        """
        MLE Vasicek calibration.

        Args:
            t_start (int): Start timestep.
            t_stop (int): Stop timestep.
            rates (npt.NDArray[np.float64]): Rates for a single instrument over time.
        """

        N = len(rates)
        dt = (t_stop-t_start)/N

        Sx = sum(rates.iloc[0:(N-1)])
        Sy = sum(rates.iloc[1:N])
        Sxx = np.dot(rates.iloc[0:(N-1)], rates.iloc[0:(N-1)])
        Sxy = np.dot(rates.iloc[0:(N-1)], rates.iloc[1:N])
        Syy = np.dot(rates.iloc[1:N], rates.iloc[1:N])
        
        theta = (Sy * Sxx - Sx * Sxy) / (N * (Sxx - Sxy) - (Sx**2 - Sx*Sy))
        kappa = -np.log((Sxy - theta * Sx - theta * Sy + N * theta**2) / (Sxx - 2*theta*Sx + N*theta**2)) / dt
        a = np.exp(-kappa * dt)
        sigmah2 = (Syy - 2*a*Sxy + a**2 * Sxx - 2*theta*(1-a)*(Sy - a*Sx) + N*theta**2 * (1-a)**2) / N
        sigma = np.sqrt(sigmah2*2*kappa / (1-a**2))
        r0 = rates.iloc[0]
        
        self.theta = theta*kappa
        self.alpha = kappa
        self.sigma = sigma
        self.r0 = r0
 