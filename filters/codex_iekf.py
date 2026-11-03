"""Numerically robust iterated extended Kalman filter (IEKF)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking.filters.filter_state import FilterState
from tracking.filters.kf_base import IKalmanFilter
from tracking.measurement_model_strategy import MeasurementModelStrategy
from tracking.models.state_model_base import IStateModel


@dataclass(frozen=True)
class IEKFSetting:
    """Configuration for :class:`IteratedExtendedKalmanFilter`.

    ``process_noise`` is passed to the state model's
    ``Q_discrete_white_noise`` method. ``convergence_tolerance`` is applied
    to the squared Mahalanobis distance between consecutive iterates.
    """

    max_iterations: int = 5
    convergence_tolerance: float = 1e-4
    process_noise: float = 0.1
    measurement_angle_indices: tuple[int, ...] = (1,)
    covariance_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive")
        if self.process_noise < 0.0:
            raise ValueError("process_noise must be non-negative")
        if self.covariance_floor <= 0.0:
            raise ValueError("covariance_floor must be positive")
        if any(index < 0 for index in self.measurement_angle_indices):
            raise ValueError("angle indices must be non-negative")


class IteratedExtendedKalmanFilter(IKalmanFilter):
    """Iterated EKF compatible with the repository's ``IKalmanFilter`` API."""

    def __init__(
        self,
        state_model: IStateModel,
        measurement_model: MeasurementModelStrategy,
        setting: IEKFSetting | None = None,
    ) -> None:
        self.state_model = state_model
        self.measurement_model = measurement_model
        self.setting = setting or IEKFSetting()

    def time_update(self, state: FilterState, dt: float) -> FilterState:
        _validate_dt(dt)
        x_prior = _as_vector(state.x, "state.x")
        P_prior = _stabilize_covariance(
            state.cov,
            len(x_prior),
            self.setting.covariance_floor,
            "state covariance",
        )

        F = np.asarray(
            self.state_model.FJacobian(x_prior, dt),
            dtype=float,
        )
        _validate_matrix(F, (len(x_prior), len(x_prior)), "state Jacobian")

        x_predicted = _as_vector(
            self.state_model.f(x_prior, dt),
            "predicted state",
        )
        if len(x_predicted) != len(x_prior):
            raise ValueError("state model changed the state dimension")

        Q = _process_covariance(
            self.state_model,
            x_prior,
            dt,
            self.setting.process_noise,
            len(x_prior),
            self.setting.covariance_floor,
        )
        P_predicted = F @ P_prior @ F.T + Q
        P_predicted = _stabilize_covariance(
            P_predicted,
            len(x_prior),
            self.setting.covariance_floor,
            "predicted covariance",
        )

        return FilterState(x=x_predicted, P=P_predicted)

    def measurement_update(
        self,
        state: FilterState,
        y: np.ndarray,
        R: np.ndarray,
    ) -> tuple[FilterState, np.ndarray, np.ndarray]:
        x_prior = _as_vector(state.x, "state.x")
        P_prior = _stabilize_covariance(
            state.cov,
            len(x_prior),
            self.setting.covariance_floor,
            "state covariance",
        )
        measurement = _as_vector(y, "measurement")
        measurement_dimension = len(measurement)
        _validate_angle_indices(
            self.setting.measurement_angle_indices,
            measurement_dimension,
        )
        R_cov = _stabilize_covariance(
            R,
            measurement_dimension,
            self.setting.covariance_floor,
            "measurement covariance",
        )

        predicted_at_prior = _measurement_vector(
            self.measurement_model.h(x_prior),
            measurement_dimension,
        )
        innovation = measurement - predicted_at_prior
        _wrap_selected(innovation, self.setting.measurement_angle_indices)

        x_iterated = x_prior.copy()
        K: np.ndarray | None = None
        H: np.ndarray | None = None

        for _ in range(self.setting.max_iterations):
            predicted_at_iterate = _measurement_vector(
                self.measurement_model.h(x_iterated),
                measurement_dimension,
            )
            H = np.asarray(
                self.measurement_model.HJacobian(x_iterated),
                dtype=float,
            )
            _validate_matrix(
                H,
                (measurement_dimension, len(x_prior)),
                "measurement Jacobian",
            )

            innovation_covariance = H @ P_prior @ H.T + R_cov
            innovation_covariance = _stabilize_covariance(
                innovation_covariance,
                measurement_dimension,
                self.setting.covariance_floor,
                "innovation covariance",
            )
            K = np.linalg.solve(
                innovation_covariance,
                H @ P_prior,
            ).T

            iterated_innovation = (
                measurement
                - predicted_at_iterate
                - H @ (x_prior - x_iterated)
            )
            _wrap_selected(
                iterated_innovation,
                self.setting.measurement_angle_indices,
            )
            x_next = x_prior + K @ iterated_innovation

            delta = x_next - x_iterated
            squared_distance = float(
                delta @ np.linalg.solve(P_prior, delta)
            )
            x_iterated = x_next
            if squared_distance <= self.setting.convergence_tolerance**2:
                break

        if K is None or H is None:  # defensive; max_iterations is validated
            raise RuntimeError("IEKF iteration did not run")

        identity = np.eye(len(x_prior))
        identity_minus_kh = identity - K @ H
        P_posterior = (
            identity_minus_kh @ P_prior @ identity_minus_kh.T
            + K @ R_cov @ K.T
        )
        P_posterior = _stabilize_covariance(
            P_posterior,
            len(x_prior),
            self.setting.covariance_floor,
            "posterior covariance",
        )

        residual = measurement - _measurement_vector(
            self.measurement_model.h(x_iterated),
            measurement_dimension,
        )
        _wrap_selected(residual, self.setting.measurement_angle_indices)

        return (
            FilterState(x=x_iterated, P=P_posterior),
            innovation,
            residual,
        )


# Short alias for callers that prefer the algorithm name.
IEKF = IteratedExtendedKalmanFilter


def _validate_dt(dt: float) -> None:
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains a non-finite value")
    return vector.copy()


def _measurement_vector(value: np.ndarray, dimension: int) -> np.ndarray:
    vector = _as_vector(value, "predicted measurement")
    if len(vector) != dimension:
        raise ValueError("measurement model returned an invalid dimension")
    return vector


def _validate_matrix(
    matrix: np.ndarray,
    shape: tuple[int, int],
    name: str,
) -> None:
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains a non-finite value")


def _stabilize_covariance(
    covariance: np.ndarray,
    dimension: int,
    floor: float,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    _validate_matrix(matrix, (dimension, dimension), name)
    matrix = 0.5 * (matrix + matrix.T)
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
    if minimum_eigenvalue < floor:
        matrix = matrix + (floor - minimum_eigenvalue) * np.eye(dimension)
    return matrix


def _process_covariance(
    state_model: IStateModel,
    x: np.ndarray,
    dt: float,
    process_noise: float,
    dimension: int,
    floor: float,
) -> np.ndarray:
    Q = state_model.Q_discrete_white_noise(x, dt, process_noise)
    return _stabilize_covariance(Q, dimension, floor, "process covariance")


def _validate_angle_indices(indices: tuple[int, ...], dimension: int) -> None:
    if any(index >= dimension for index in indices):
        raise ValueError("an angle index is outside the measurement vector")


def _wrap_selected(vector: np.ndarray, indices: tuple[int, ...]) -> None:
    for index in indices:
        vector[index] = (vector[index] + np.pi) % (2.0 * np.pi) - np.pi
