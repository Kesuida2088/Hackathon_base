import numpy as np

from tracking.filters.kf_base import FilterState, IKalmanFilter
from tracking.measurement_model_strategy import MeasurementModelStrategy
from tracking.models.state_model_base import IStateModel
from tracking.sigma_points import CubatureSigmaPoints


class CubatureKalmanFilter(IKalmanFilter):
    def __init__(
        self,
        state_model: IStateModel,
        measurement_model: MeasurementModelStrategy,
    ):
        self.state_model = state_model
        self.measurement_model = measurement_model
        self.cubature_points = CubatureSigmaPoints()

    def time_update(
        self,
        state: FilterState,
        dt: float,
    ) -> FilterState:
        cubs, W = self.cubature_points.generate(state)

        cubs_f = np.zeros_like(cubs)
        for i in range(len(cubs)):
            cubs_f[i] = self.state_model.f(cubs[i], dt)

        Q = self.state_model.Q_discrete_white_noise(
            state.x,
            dt,
            process_noise=0.0001,
        )
        x, P = self._cubature_transform(cubs_f, W, Q)

        return FilterState(x=x, P=P)

    def measurement_update(
        self,
        state: FilterState,
        y: np.array,
        R: np.ndarray,
    ) -> tuple[FilterState, np.ndarray, np.ndarray]:
        cubs, W = self.cubature_points.generate(state)

        cubs_h = np.zeros((cubs.shape[0], len(y)))
        for i in range(len(cubs)):
            cubs_h[i] = self.measurement_model.h(cubs[i])

        xz, Pz = self._cubature_transform(
            cubs_h,
            W,
            R,
            angle_index=1,
        )

        Pxz = np.zeros((len(state.x), len(xz)))
        for i in range(len(cubs)):
            dx = cubs[i] - state.x
            dy = cubs_h[i] - xz
            dy[1] = self.wrap_angle(dy[1])
            Pxz += W[i] * np.outer(dx, dy)

        K = Pxz @ np.linalg.solve(Pz, np.eye(Pz.shape[0]))

        innovation = y - xz
        innovation[1] = self.wrap_angle(innovation[1])

        x = state.x + K @ innovation
        P = state.P - K @ Pz @ K.T
        P = 0.5 * (P + P.T)

        residual = y - self.measurement_model.h(x)
        residual[1] = self.wrap_angle(residual[1])

        return FilterState(x=x, P=P), innovation, residual

    def _cubature_transform(
        self,
        cubs: np.ndarray,
        W: np.ndarray,
        noise_cov: np.ndarray,
        angle_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = W @ cubs
        if angle_index is not None:
            x[angle_index] = np.arctan2(
                W @ np.sin(cubs[:, angle_index]),
                W @ np.cos(cubs[:, angle_index]),
            )

        P = np.zeros((len(x), len(x)))
        for i in range(len(cubs)):
            dx = cubs[i] - x
            if angle_index is not None:
                dx[angle_index] = self.wrap_angle(dx[angle_index])
            P += W[i] * np.outer(dx, dx)

        P += noise_cov
        P = 0.5 * (P + P.T)

        return x, P
