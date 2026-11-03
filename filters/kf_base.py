import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from scipy.linalg import solve_triangular
from .filter_state import FilterState


class FilterType(Enum):
    EKF = auto()  # Extended Kalman Filter
    UKF = auto()  # Unscented Kalman Filter
    CKF = auto()  # Cubature Kalman Filter
    SREKF = auto()  # Square-Root Extended Kalman Filter
    SRUKF = auto()  # Square-Root Unscented Kalman Filter
    SRCKF = auto()  # Square-Root Cubature Kalman Filter


class IKalmanFilter(ABC):
    @abstractmethod
    def time_update(self, state: FilterState, dt: float):
        ...

    @abstractmethod
    def measurement_update(
        self,
        state: FilterState,
        y: np.array,
        R: np.ndarray
    ):
        ...

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


class ISquareRootKalmanFilter(ABC):
    # Givens Rotationアルゴリズム
    @staticmethod
    def cholupdate(L: np.ndarray, x: np.array) -> np.ndarray:
        # 正定値性チェック
        try:
            p = solve_triangular(L, x, lower=True)
            if np.sum(p**2) >= 1.0:
                raise ValueError("L is not positive definite matrix")
        except np.linalg.LinAlgError:
            raise ValueError("L is singular matrix")

        n = L.shape[0]
        L_new = L.copy()
        x_new = x.copy()

        for i in range(n):
            # 対角要素計算
            r = np.sqrt(L_new[i, i]**2 + x_new[i]**2)
            L_new[i, i] = r

            # 非対角要素計算パラメータ
            cos = L[i, i] / r
            sin = x_new[i] / r

            # 非対角要素&ベクトルx更新
            if i+1 < n:
                # 列ベクトル抽出
                L_col = L_new[i+1:, i].copy()
                # Lの下側要素更新
                L_new[i+1:, i] = cos * L_col + sin * x_new[i+1:]
                # xの更新
                x_new[i+1:] = cos * x_new[i+1:] - sin * L_col

        return L_new

    @staticmethod
    def choldowndate(L: np.ndarray, x: np.array) -> np.ndarray:
        n = L.shape[0]
        L_new = L.copy()
        x_new = x.copy()

        for i in range(n):
            # 対角要素計算
            r = np.sqrt(L_new[i, i]**2 + x_new[i]**2)
            L_new[i, i] = r

            # 非対角要素計算パラメータ
            cos = r / L[i, i]
            sin = r / x_new[i]

            # 非対角要素&ベクトルx更新
            if i+1 < n:
                # 列ベクトル抽出
                L_col = L_new[i+1:, i].copy()
                # Lの下側要素更新
                L_new[i+1:, i] = cos * L_col + sin * x_new[i+1:]
                # xの更新
                x_new[i+1:] = cos * x_new[i+1:] - sin * L_col

        return L_new
