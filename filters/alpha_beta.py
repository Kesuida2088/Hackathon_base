import numpy as np

from tracking.filters.kf_base import FilterState, IKalmanFilter


class AlphaBetaFilter(IKalmanFilter):
    def __init__(
        self,
        alpha: float,
        beta: float,
        radar_pos: np.ndarray,
        default_dt: float,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not 0.0 <= beta <= 2.0:
            raise ValueError("beta must be in [0, 2]")
        if default_dt <= 0.0:
            raise ValueError("default_dt must be positive")

        self.alpha = alpha
        self.beta = beta
        self.radar_pos = np.asarray(radar_pos, dtype=float)
        self._dt = default_dt

    def time_update(self, state: FilterState, dt: float) -> FilterState:
        self._dt = dt
        x = state.x.copy()
        x[0:2] += x[2:4] * dt
        return FilterState(x=x, P=state.P)

    def measurement_update(
        self,
        state: FilterState,
        y: np.array,
        R: np.ndarray,
    ) -> tuple[FilterState, np.ndarray, np.ndarray]:
        z = self._measurement_to_cartesian(y)
        residual = z - state.x[0:2]

        x = state.x.copy()
        x[0:2] += self.alpha * residual
        x[2:4] += (self.beta / self._dt) * residual

        P = state.P.copy() if state.P is not None else np.eye(len(x))
        innovation = residual.copy()
        post_residual = z - x[0:2]

        return FilterState(x=x, P=P), innovation, post_residual

    def _measurement_to_cartesian(self, measurement: np.ndarray) -> np.ndarray:
        rng, azimuth = measurement
        return self.radar_pos + np.array([
            rng * np.cos(azimuth),
            rng * np.sin(azimuth),
        ])
