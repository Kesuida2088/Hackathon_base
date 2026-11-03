from dataclasses import dataclass

import numpy as np
from numpy import eye
from numpy.linalg import cholesky, norm, solve
from scipy.linalg import solve_triangular

from tracking.filters.kf_base import FilterState, IKalmanFilter
from tracking.measurement_model_strategy import MeasurementModelStrategy
from tracking.models.state_model_base import IStateModel


@dataclass(frozen=True)
class EKFSetting:
    ite_max: int
    ite_eps: float

    def __post_init__(self):
        if self.ite_max < 1:
            raise ValueError("Max iteration number must be greater than 1")

        if self.ite_eps <= 0.0:
            raise ValueError("Iteration epsilon must be greater than 0.0")


class SquareRootExtendedKalmanFilter(IKalmanFilter):
    def __init__(
        self,
        state_model: IStateModel,
        measurement_model: MeasurementModelStrategy,
        setting: EKFSetting,
    ):
        # 状態モデル
        self.state_model = state_model

        # 観測モデル
        self.measurement_model = measurement_model

        # EKF設定
        self.setting = setting

    # 時間更新
    def time_update(self, state: FilterState, dt: float):
        F = self.state_model.FJacobian(state.x, dt)
        Q = self.state_model.Q_discrete_white_noise(
            state.x,
            dt,
            process_noise=0.01,
        )
        Q = 0.5 * (Q + Q.T) + 1e-12 * eye(Q.shape[0])

        x = self.state_model.f(state.x, dt)
        P = F @ state.cov @ F.T + Q

        return FilterState(x=x, S=_safe_cholesky(P))

    def measurement_update(
        self,
        state: FilterState,
        y: np.array,
        SR: np.ndarray,
    ) -> FilterState:
        h = self.measurement_model.h
        HJacobian = self.measurement_model.HJacobian

        P_prior = state.cov
        R_cov = SR @ SR.T
        x_ite = state.x.copy()

        for _ in range(self.setting.ite_max):
            H = HJacobian(x_ite)
            innovation_cov = H @ P_prior @ H.T + R_cov
            K = P_prior @ H.T @ solve(innovation_cov, eye(len(y)))

            innovation = y - h(x_ite) - H @ (state.x - x_ite)
            innovation[1] = self.wrap_angle(innovation[1])

            x = state.x + K @ innovation
            delta = x - x_ite

            try:
                d = solve_triangular(state.S, delta, lower=True)
                distance = np.sum(d**2)
                if distance < self.setting.ite_eps:
                    break
            except np.linalg.LinAlgError:
                if norm(delta) < self.setting.ite_eps:
                    break

            x_ite = x

        I_KH = eye(len(state.x)) - K @ H
        P = I_KH @ P_prior @ I_KH.T + K @ R_cov @ K.T

        res = y - h(x)
        res[1] = self.wrap_angle(res[1])

        return FilterState(x=x, S=_safe_cholesky(P)), innovation, res


def _safe_cholesky(P: np.ndarray) -> np.ndarray:
    P = 0.5 * (P + P.T)
    jitter = 1e-12

    for _ in range(6):
        try:
            return cholesky(P)
        except np.linalg.LinAlgError:
            P = P + jitter * eye(P.shape[0])
            jitter *= 10.0

    return cholesky(P)
