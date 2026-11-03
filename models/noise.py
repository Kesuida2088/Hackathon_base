from abc import ABC, abstractmethod
import numpy as np


class INoiseModel(ABC):
    @abstractmethod
    def cov(self, x, dt) -> np.ndarray:
        pass

    @abstractmethod
    def sqrt_cov(self, x, dt) -> np.ndarray:
        pass


class IMeasurementNoiseModel(ABC):
    @abstractmethod
    def cov(self, x) -> np.ndarray:
        pass

    @abstractmethod
    def sqrt_cov(self, x) -> np.ndarray:
        pass


class NCVProcessNoise(INoiseModel):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def cov(self, x, dt) -> np.ndarray:
        q = self.sigma **2

    def sqrt_cov(self, x, dt) -> np.ndarray:
        pass


class NCAProcessNoise(INoiseModel):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def cov(self, x, dt) -> np.ndarray:
        pass

    def sqrt_cov(self, x, dt) -> np.ndarray:
        pass


class NCTProcessNoise(INoiseModel):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def cov(self, x, dt) -> np.ndarray:
        pass

    def sqrt_cov(self, x, dt) -> np.ndarray:
        pass
