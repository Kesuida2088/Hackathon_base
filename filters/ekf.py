from tracking.models.state_model_base import IStateModel
from tracking.measurement_model_strategy import MeasurementModelStrategy
from tracking.filters.kf_base import IKalmanFilter, FilterState
import numpy as np
from numpy.linalg import norm, solve
from numpy import eye
from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class EKFSetting:
    ite_max: int         # 反復最大回数
    ite_eps: float       # 反復精度

    def __post_init__(self):
        if self.ite_max < 1:
            raise ValueError("Max iteration number must be greater than 1")

        if self.ite_eps <= 0.0:
            raise ValueError("Iteration epsilon must be greater than 0.0")


class IterationMethod(Enum):
    NONE = auto()    # 通常EKF(反復なし)
    IEKF = auto()    # 線形化を反復


# 収束判定
class ConvergenceMetric(Enum):
    EUCLIDEAN = auto()  # ユークリッド距離
    MAHALANOBIS = auto()  # マハラノビス距離


@dataclass(frozen=True)
class IterationSetting:
    max_ite: int
    eps: float
    method: IterationMethod
    metric: ConvergenceMetric

    def __post_init__(self):
        if self.max_ite < 1:
            raise ValueError("Max iteration number must be greater than 1")

        if self.eps <= 0.0:
            raise ValueError("Iteration epsilon must be greater than 0.0")


class EKFConvergence:
    def __init__(self, eps: float):
        self.eps = eps

    def has_converge(
        self,
        delta: np.array
    ) -> bool:
        return norm(delta) < self.eps


class IEKFIterationPolicy:
    def __init__(self, setting: IterationSetting):
        self.setting = setting

    @property
    def max_iter(self):
        if self.setting.method == IterationMethod.NONE:
            return 1
        return self.setting.max_ite

    def has_converge(
        self,
        delta: np.array
    ) -> bool:
        if self.setting.method == IterationMethod.NONE:
            return True

        if self.setting.metric == ConvergenceMetric.EUCLIDEAN:
            return norm(delta) < self.setting.eps

        if self.setting.metric == ConvergenceMetric.MAHALANOBIS:
            return False


class ExtendedKalmanFilter(IKalmanFilter):
    def __init__(
        self,
        state_model: IStateModel,
        measurement_model: MeasurementModelStrategy,
        setting: EKFSetting
    ):
        # 状態モデル
        self.state_model = state_model
        # 観測モデル
        self.measurement_model = measurement_model
        # EKF設定
        self.setting = setting

    def time_update(self, state: FilterState, dt: float):
        # 状態遷移行列/プロセスノイズ
        F = self.state_model.FJacobian(state.x, dt)
        # Q = self.state_model.Q_continuous_white_noise(state.x, dt, process_noise=0.04)
        Q = self.state_model.Q_discrete_white_noise(state.x, dt, process_noise=0.1)
        # 1段予測値
        x = self.state_model.f(state.x, dt)
        # 1段予測値の推定誤差共分散行列
        P = F @ state.P @ F.T + Q

        return FilterState(x, P)

    # TODO
    # IEKF部分見直し、SR-IEKFの方法参照
    # 繰り返し完了をマハラノビス距離で
    def measurement_update(
        self,
        state: FilterState,
        y: np.array,
        R: np.ndarray
    ) -> FilterState:
        h = self.measurement_model.h
        HJacobian = self.measurement_model.HJacobian

        x_ite = state.x.copy()

        # 濾波推定値近傍でh(x)の線形化を繰り返し推定精度を高める(IEKF)
        for _ in range(self.setting.ite_max):
            # 観測行列
            H = HJacobian(x_ite)

            # S=(HP(t|t-1)H'+R)^-1
            # np.linalg.inv()を使用するより高速
            S = solve(H @ state.P @ H.T + R, eye(len(y)))

            # カルマンゲイン
            K = state.P @ H.T @ S

            # 濾波推定値
            innovation = y - h(x_ite) - H @ (state.x - x_ite)
            innovation[1] = self.wrap_angle(innovation[1])
            x = state.x + K @ innovation

            # 収束判定
            if norm(x - x_ite) < self.setting.ite_eps:
                break
            x_ite = x

        # 濾波推定誤差共分散行列
        # Josephの安定化アルゴリズム P(t|t)=(I-KH)P(t|t-1)(I-KH)'+KP(t|t-1)K'を用いると
        # 行列の対称性が保たれ数値安定が高まる
        # 一般式はP=(I-KH)P
        I_KH = eye(len(state.x)) - K @ H
        P = I_KH @ state.P @ I_KH.T + K @ R @ K.T

        # イノベーション
        z = y - h(state.x)
        z[1] = self.wrap_angle(z[1])

        # 残差
        res = y - h(x)
        res[1] = self.wrap_angle(res[1])

        return FilterState(x, P), z, res
