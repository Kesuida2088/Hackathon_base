from abc import ABC, abstractmethod
import numpy as np
from .filters.kf_base import FilterState


class SigmaPointsStrategy(ABC):
    @abstractmethod
    def generate(self, state: FilterState):
        pass


class MerweScaledSigmaPoints(SigmaPointsStrategy):
    def __init__(self, alpha, beta, kappa):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def generate(self, state: FilterState):
        n = len(state.x)
        lam = self.alpha**2 * (n + self.kappa) - n

        # コレスキー分解で平方根行列を求める
        S = np.linalg.cholesky((n + lam) * state.P)

        # シグマポイント生成
        sigma = np.zeros((2*n+1, n))
        sigma[0] = state.x

        for i in range(n):
            sigma[i+1] = state.x + S[:, i]
            sigma[n+i+1] = state.x - S[:, i]

        # シグマポイント（平均）の重み
        Wm = np.full(2*n+1, 1.0 / (2 * (n + lam)))
        Wm[0] = lam / (n + lam)

        # シグマポイント（共分散行列）の重み
        Wc = np.full(2*n+1, 1.0 / (2 * (n + lam)))
        Wc[0] = lam / (n + lam) + 1 - self.alpha**2 + self.beta

        return sigma, Wm, Wc


class SquareRootMerweScaledSigmaPoints(SigmaPointsStrategy):
    def __init__(self, alpha, beta, kappa):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def generate(self, state: FilterState):
        n = len(state.x)
        lam = self.alpha**2 * (n + self.kappa) - n
        gamma = np.sqrt(n + lam)

        # シグマポイント生成
        sigma = np.zeros((2*n+1, n))
        sigma[0] = state.x

        for i in range(n):
            sigma[i+1] = state.x + gamma * state.S[:, i]
            sigma[n+i+1] = state.x - gamma * state.S[:, i]

        # シグマポイント（平均）の重み
        Wm = np.full(2*n+1, 1.0 / (2 * (n + lam)))
        Wm[0] = lam / (n + lam)

        # シグマポイント（共分散行列）の重み
        Wc = np.full(2*n+1, 1.0 / (2 * (n + lam)))
        Wc[0] = lam/(n + lam) + 1 - self.alpha**2 + self.beta

        return sigma, Wm, Wc


class CubatureSigmaPoints(SigmaPointsStrategy):
    def generate(self, state: FilterState):
        n = len(state.x)
        offsets = (np.sqrt(n) * np.linalg.cholesky(state.P)).T

        points = np.empty((2 * n, n))
        points[:n] = state.x + offsets
        points[n:] = state.x - offsets

        weights = np.full(2 * n, 1.0 / (2 * n))

        return points, weights
