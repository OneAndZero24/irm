from abc import ABCMeta, abstractmethod

import pandas as pd


class IRModel(metaclass=ABCMeta):
    """
    Abstract base class for an Interest Rate Model with following form:
    dY = a(Y, t) dt + b(Y, t) dW

    Methods:
    a(Y_prev: float, t: float) -> float
        a(Y, t)

    b(Y_prev: float, t: float) -> float
        b(Y, t)

    Y0() -> float
        Starting point for chain.

    differentiable -> bool
        Needed for Milstein.

    calibrate(data: pd.DataFrame)
        Abstract method to calibrate the model using the provided data.
    """

    @abstractmethod
    def a(self, Y_prev: float, t: float) -> float:
        pass

    @abstractmethod
    def b(self, Y_prev: float, t: float) -> float:
        pass

    @abstractmethod
    def Y0(self) -> float:
        pass

    @abstractmethod
    def differentiable(self) -> bool:
        pass


    @abstractmethod
    def calibrate(self, data: pd.DataFrame):
        pass
