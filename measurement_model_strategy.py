from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import norm
from math import atan2


class MeasurementModelStrategy(ABC):
    @abstractmethod
    def h(self, x):
        pass

    @abstractmethod
    def HJacobian(self, x):
        pass


class NearlyConstantVelocity(MeasurementModelStrategy):
    def __init__(self, radar_pos: np.array):
        if len(radar_pos) != 2:
            raise ValueError("Radar position must be 2 dimensions")

        self.radar_pos = radar_pos

    def h(self, x):
        if len(x) != 4:
            raise ValueError("State vector must be 4 dimensions")

        dx = x[0] - self.radar_pos[0]
        dy = x[1] - self.radar_pos[1]
        return np.array([norm(x[0:2]-self.radar_pos), atan2(dy, dx)])

    def HJacobian(self, x):
        if len(x) != 4:
            raise ValueError("State vector must be 4 dimensions")

        r = norm(x[0:2] - self.radar_pos)
        dx = x[0] - self.radar_pos[0]
        dy = x[1] - self.radar_pos[1]

        H = np.array([
            [dx/r    , dy/r   , 0.0, 0.0],
            [-dy/r**2, dx/r**2, 0.0, 0.0]
        ])
        return H


class NearlyConstantAcceleration(MeasurementModelStrategy):
    def __init__(self, radar_pos: np.array):
        if len(radar_pos) != 2:
            raise ValueError("Radar position must be 2 dimensions")

        self.radar_pos = radar_pos

    def h(self, x):
        if len(x) != 6:
            raise ValueError("State vector must be 4 dimensions")

        dx = x[0] - self.radar_pos[0]
        dy = x[1] - self.radar_pos[1]
        return np.array([norm(x[0:2]-self.radar_pos), atan2(dy, dx)])

    def HJacobian(self, x):
        if len(x) != 6:
            raise ValueError("State vector must be 4 dimensions")

        r = norm(x[0:2] - self.radar_pos)
        dx = x[0] - self.radar_pos[0]
        dy = x[1] - self.radar_pos[1]

        H = np.array([
            [dx/r    , dy/r   , 0.0, 0.0, 0.0, 0.0],
            [-dy/r**2, dx/r**2, 0.0, 0.0, 0.0, 0.0]
        ])
        return H


class NearlyConstantJerk(MeasurementModelStrategy):
    def __init__(self, radar_pos: np.array):
        if len(radar_pos) != 2:
            raise ValueError("Radar position must be 2 dimensions")

        self.radar_pos = radar_pos

    def h(self, x):
        if len(x) != 8:
            raise ValueError("State vector must be 8 dimensions")

        dx = x[0] - self.radar_pos[0]
        dy = x[1] - self.radar_pos[1]
        return np.array([norm(x[0:2]-self.radar_pos), atan2(dy, dx)])

    def HJacobian(self, x):
        if len(x) != 8:
            raise ValueError("State vector must be 8 dimensions")

        r = norm(x[0:2] - self.radar_pos)
        dx = x[0] - self.radar_pos[0]
        dy = x[1] - self.radar_pos[1]

        H = np.array([
            [dx/r    , dy/r   , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-dy/r**2, dx/r**2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        return H


class NearlyConstantTurn(MeasurementModelStrategy):
    def __init__(self, radar_pos: np.array):
        if len(radar_pos) != 2:
            raise ValueError("Radar position must be 2 dimensions")

        self.radar_pos = radar_pos

    def h(self, x):
        if len(x) != 5:
            raise ValueError("State vector must be 8 dimensions")

        dx = x[0] - self.radar_pos[0]
        dy = x[1] - self.radar_pos[1]
        return np.array([norm(x[0:2]-self.radar_pos), atan2(dy, dx)])

    def HJacobian(self, x):
        if len(x) != 5:
            raise ValueError("State vector must be 8 dimensions")

        r = norm(x[0:2] - self.radar_pos)
        dx = x[0] - self.radar_pos[0]
        dy = x[1] - self.radar_pos[1]

        H = np.array([
            [dx/r    , dy/r   , 0.0, 0.0, 0.0],
            [-dy/r**2, dx/r**2, 0.0, 0.0, 0.0]
        ])
        return H
