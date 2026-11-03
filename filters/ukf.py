from tracking.filters.kf_base import IKalmanFilter, FilterState
from tracking.models.state_model_base import IStateModel
from tracking.measurement_model_strategy import MeasurementModelStrategy
from tracking.sigma_points import SigmaPointsStrategy
import numpy as np


class UnscentedKalmanFilter(IKalmanFilter):
    def __init__(
        self,
        state_model: IStateModel,
        measurement_model: MeasurementModelStrategy,
        sigma_points_strategy: SigmaPointsStrategy
    ):
        # 状態モデル
        self.state_model = state_model

        # 観測モデル
        self.measurement_model = measurement_model

        # シグマポイント選択アルゴリズム
        self.sigma_points = sigma_points_strategy

    def time_update(self, state: FilterState, dt: float):
        # シグマポイントと重み生成
        sigmas, Wm, Wc = self.sigma_points.generate(state)

        # シグマポイントを状態モデルに通す
        sigmas_f = np.zeros_like(sigmas)
        for i in range(len(sigmas)):
            sigmas_f[i] = self.state_model.f(sigmas[i], dt)

        # プロセスノイズ
        # Q = self.state_model.Q_continuous_white_noise(state.x, dt, process_noise=0.07)
        Q = self.state_model.Q_discrete_white_noise(state.x, dt, process_noise=0.0001)

        # U変換で1段予測値と1段予測誤差共分散行列を求める
        state_pre = self._unscented_transform(sigmas_f, Wm, Wc, Q)

        return state_pre

    def measurement_update(
        self, state: FilterState,
        y: np.array,
        R: np.ndarray
    ):
        # シグマポイントと重み生成
        sigmas, Wm, Wc = self.sigma_points.generate(state)

        # シグマポイントを観測モデルに通す
        sigmas_h = np.zeros((sigmas.shape[0], len(y)))
        for i in range(len(sigmas)):
            sigmas_h[i] = self.measurement_model.h(sigmas[i])

        # U変換で平均と共分散行列を求める
        measurement_state = self._unscented_transform(
            sigmas_h,
            Wm,
            Wc,
            R,
            angle_index=1,
        )
        xz = measurement_state.x
        Pz = measurement_state.P

        Pxz = np.zeros((len(state.x), len(xz)))
        for i in range(len(sigmas)):
            dx = sigmas[i] - state.x
            dy = sigmas_h[i] - xz
            dy[1] = self.wrap_angle(dy[1])
            Pxz += Wc[i] * np.outer(dx, dy)

        # カルマンゲイン
        K = Pxz @ np.linalg.solve(Pz, np.eye(Pz.shape[0]))

        # イノベーション
        z = y - xz
        z[1] = self.wrap_angle(z[1])

        # 濾波推定値
        x = state.x + K @ z

        # 濾波推定誤差共分散行列
        P = state.P - K @ Pz @ K.T

        # 残差
        res = y - self.measurement_model.h(x)
        res[1] = self.wrap_angle(res[1])

        return FilterState(x, P), z, res

    def _unscented_transform(
        self,
        sigmas: np.ndarray,
        Wm: np.ndarray,
        Wc: np.ndarray,
        R: np.ndarray,
        angle_index: int | None = None
    ):
        # 重み付き平均
        x = Wm @ sigmas
        if angle_index is not None:
            x[angle_index] = np.arctan2(
                Wm @ np.sin(sigmas[:, angle_index]),
                Wm @ np.cos(sigmas[:, angle_index]),
            )

        # 重み付き共分散行列
        P = np.zeros((len(x), len(x)))
        for i in range(len(sigmas)):
            dx = sigmas[i] - x
            if angle_index is not None:
                dx[angle_index] = self.wrap_angle(dx[angle_index])
            P += Wc[i] * np.outer(dx, dx)

        P += R

        P = 0.5 * (P + P.T)

        return FilterState(x, P)
