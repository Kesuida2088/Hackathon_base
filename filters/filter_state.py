from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FilterState:
    x: np.array                     # 状態変数
    P: Optional[np.ndarray] = None  # 誤差共分散行列
    S: Optional[np.ndarray] = None  # 誤差共分散行列（平方根）

    def __post_init__(self):
        if self.P is not None and self.S is not None:
            raise ValueError("P and S cannot be set at the same time")

        if self.P is None and self.S is None:
            raise ValueError("P or S must be set")

        if self.P is not None:
            if self.P.shape[0] != self.P.shape[1]:
                raise ValueError("P must be a square matrix")
            if self.x.shape[0] != self.P.shape[0]:
                raise ValueError("x and P must have the same dimension")

        if self.S is not None:
            if self.S.shape[0] != self.S.shape[1]:
                raise ValueError("S must be a square matrix")
            if self.x.shape[0] != self.S.shape[0]:
                raise ValueError("x and S must have the same dimension")

    @property
    def cov(self) -> np.ndarray:
        if self.P is not None:
            return self.P
        elif self.S is not None:
            return self.S @ self.S.T
        else:
            raise ValueError("P or S must be set")


@dataclass(frozen=True)
class FilterStatistic:
    K:          np.ndarray  # カルマンゲイン
    innovation: np.array    # イノベーション
    S:          np.ndarray  # イノベーションの誤差共分散行列
    residual:   np.array    # 残差

    def __post_init__(self):
        if self.S.shape[0] != self.S.shape[1]:
            raise ValueError("S must be a square matrix")

        if self.innovation.shape[0] != self.S.shape[0]:
            raise ValueError("innovation and S must have the same dimension")


@dataclass(frozen=True)
class EpochFilterState:
    t:          int
    state_pre:  FilterState
    state_post: FilterState
    statistic:  FilterStatistic

    def __post_init__(self):
        if self.t < 0:
            raise ValueError("t must be a non-negative integer")
