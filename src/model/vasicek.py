from .ir_model import IRModel

import pandas as pd


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

    Methods:
        a(Y_prev: float) -> float:
            Computes the drift term of the Vasicek model.
        b(Y_prev: float) -> float:
            Computes the diffusion term of the Vasicek model.
        calibrate(data: pd.DataFrame):
            Calibrates the model parameters to market data.
    """

    def __init__(self, theta: float, alpha: float, sigma: float):
        """
        Initializes the Vasicek model with the given parameters.

        Args:
            theta (float): The long-term mean level of the interest rate.
            alpha (float): The speed of reversion to the mean.
            sigma (float): The volatility of the interest rate.
        """

        super().__init__()
        self.theta = theta
        self.alpha = alpha
        self.sigma = sigma
     
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
    
    def calibrate(self, data: pd.DataFrame):
        """
        Calibrates the model parameters to market data.
        Args:
            data (pd.DataFrame): The market data to calibrate the model to.
        """

        pass
