from abc import ABCMeta, abstractmethod

import pandas as pd


class IRModel(metaclass=ABCMeta):
    """
    Abstract base class for an Interest Rate Model with following form:
    dY = a(Y) dt + b(Y) dW

    Methods:
    a(t: float) -> float
        a(Y)

    b(t: float) -> float
        b(Y)

    calibrate(data: pd.DataFrame)
        Abstract method to calibrate the model using the provided data.
    """

    @abstractmethod
    def a(self, Y_prev: float) -> float:
        pass

    @abstractmethod
    def b(self, Y_prev: float) -> float:
        pass

    @abstractmethod
    def calibrate(self, data: pd.DataFrame):
        pass
