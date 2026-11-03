"""Scaled unscented Kalman filter with configurable circular components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking.filters.filter_state import FilterState
from tracking.filters.kf_base import IKalmanFilter
from tracking.measurement_model_strategy import MeasurementModelStrategy
from tracking.models.state_model_base import IStateModel


@dataclass(frozen=True)
class UKFSetting:
    alpha: float = 0.3
    beta: float = 2.0
    kappa: float = 0.0
    process_noise: float = 0.1
    state_angle_indices: tuple[int, ...] = ()
    measurement_angle_indices: tuple[int, ...] = (1,)
    covariance_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if self.process_noise < 0.0:
            raise ValueError("process_noise must be non-negative")
        if self.covariance_floor <= 0.0:
            raise ValueError("covariance_floor must be positive")
        indices = self.state_angle_indices + self.measurement_angle_indices
        if any(index < 0 for index in indices):
            raise ValueError("angle indices must be non-negative")


class UnscentedKalmanFilter(IKalmanFilter):
    """Scaled UKF using Merwe sigma points.

    The class accepts either ``P`` or ``S`` input states through
    :attr:`FilterState.cov` and returns a covariance-form state.
    """

    def __init__(
        self,
        state_model: IStateModel,
        measurement_model: MeasurementModelStrategy,
        setting: UKFSetting | None = None,
    ) -> None:
        self.state_model = state_model
        self.measurement_model = measurement_model
        self.setting = setting or UKFSetting()

    def time_update(self, state: FilterState, dt: float) -> FilterState:
        _validate_dt(dt)
        x = _as_vector(state.x, "state.x")
        dimension = len(x)
        _validate_angle_indices(self.setting.state_angle_indices, dimension)
        P = _stabilize_covariance(
            state.cov,
            dimension,
            self.setting.covariance_floor,
            "state covariance",
        )

        sigma_points, mean_weights, covariance_weights = self._sigma_points(x, P)
        propagated = np.asarray(
            [self.state_model.f(point, dt) for point in sigma_points],
            dtype=float,
        )
        _validate_point_matrix(
            propagated,
            len(sigma_points),
            dimension,
            "propagated sigma points",
        )

        predicted_mean = _weighted_mean(
            propagated,
            mean_weights,
            self.setting.state_angle_indices,
        )
        Q = _process_covariance(
            self.state_model,
            x,
            dt,
            self.setting.process_noise,
            dimension,
            self.setting.covariance_floor,
        )
        predicted_covariance = _weighted_covariance(
            propagated,
            predicted_mean,
            covariance_weights,
            self.setting.state_angle_indices,
        ) + Q
        predicted_covariance = _stabilize_covariance(
            predicted_covariance,
            dimension,
            self.setting.covariance_floor,
            "predicted covariance",
        )

        return FilterState(x=predicted_mean, P=predicted_covariance)

    def measurement_update(
        self,
        state: FilterState,
        y: np.ndarray,
        R: np.ndarray,
    ) -> tuple[FilterState, np.ndarray, np.ndarray]:
        x = _as_vector(state.x, "state.x")
        state_dimension = len(x)
        _validate_angle_indices(self.setting.state_angle_indices, state_dimension)
        P = _stabilize_covariance(
            state.cov,
            state_dimension,
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

        sigma_points, mean_weights, covariance_weights = self._sigma_points(x, P)
        measurement_points = np.asarray(
            [self.measurement_model.h(point) for point in sigma_points],
            dtype=float,
        )
        _validate_point_matrix(
            measurement_points,
            len(sigma_points),
            measurement_dimension,
            "measurement sigma points",
        )

        predicted_measurement = _weighted_mean(
            measurement_points,
            mean_weights,
            self.setting.measurement_angle_indices,
        )
        innovation_covariance = _weighted_covariance(
            measurement_points,
            predicted_measurement,
            covariance_weights,
            self.setting.measurement_angle_indices,
        ) + R_cov
        innovation_covariance = _stabilize_covariance(
            innovation_covariance,
            measurement_dimension,
            self.setting.covariance_floor,
            "innovation covariance",
        )

        cross_covariance = np.zeros((state_dimension, measurement_dimension))
        for point, measurement_point, weight in zip(
            sigma_points,
            measurement_points,
            covariance_weights,
        ):
            state_difference = point - x
            measurement_difference = measurement_point - predicted_measurement
            _wrap_selected(state_difference, self.setting.state_angle_indices)
            _wrap_selected(
                measurement_difference,
                self.setting.measurement_angle_indices,
            )
            cross_covariance += weight * np.outer(
                state_difference,
                measurement_difference,
            )

        gain = np.linalg.solve(
            innovation_covariance,
            cross_covariance.T,
        ).T
        innovation = measurement - predicted_measurement
        _wrap_selected(innovation, self.setting.measurement_angle_indices)

        posterior_mean = x + gain @ innovation
        _wrap_selected(posterior_mean, self.setting.state_angle_indices)
        posterior_covariance = P - gain @ innovation_covariance @ gain.T
        posterior_covariance = _stabilize_covariance(
            posterior_covariance,
            state_dimension,
            self.setting.covariance_floor,
            "posterior covariance",
        )

        residual = measurement - _measurement_vector(
            self.measurement_model.h(posterior_mean),
            measurement_dimension,
        )
        _wrap_selected(residual, self.setting.measurement_angle_indices)

        return (
            FilterState(x=posterior_mean, P=posterior_covariance),
            innovation,
            residual,
        )

    def _sigma_points(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dimension = len(mean)
        scaling_parameter = (
            self.setting.alpha**2 * (dimension + self.setting.kappa)
            - dimension
        )
        scale = dimension + scaling_parameter
        if scale <= 0.0:
            raise ValueError("UKF parameters require n + lambda to be positive")

        root = _safe_cholesky(
            scale * covariance,
            self.setting.covariance_floor,
        )
        sigma_points = np.empty((2 * dimension + 1, dimension))
        sigma_points[0] = mean
        for index in range(dimension):
            offset = root[:, index]
            sigma_points[index + 1] = mean + offset
            sigma_points[dimension + index + 1] = mean - offset

        mean_weights = np.full(2 * dimension + 1, 0.5 / scale)
        covariance_weights = mean_weights.copy()
        mean_weights[0] = scaling_parameter / scale
        covariance_weights[0] = (
            mean_weights[0]
            + 1.0
            - self.setting.alpha**2
            + self.setting.beta
        )
        return sigma_points, mean_weights, covariance_weights


UKF = UnscentedKalmanFilter


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


def _validate_point_matrix(
    points: np.ndarray,
    point_count: int,
    dimension: int,
    name: str,
) -> None:
    if points.shape != (point_count, dimension):
        raise ValueError(
            f"{name} must have shape {(point_count, dimension)}, got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains a non-finite value")


def _stabilize_covariance(
    covariance: np.ndarray,
    dimension: int,
    floor: float,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must have shape {(dimension, dimension)}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains a non-finite value")
    matrix = 0.5 * (matrix + matrix.T)
    minimum_eigenvalue = float(np.linalg.eigvalsh(matrix)[0])
    if minimum_eigenvalue < floor:
        matrix = matrix + (floor - minimum_eigenvalue) * np.eye(dimension)
    return matrix


def _safe_cholesky(covariance: np.ndarray, floor: float) -> np.ndarray:
    dimension = covariance.shape[0]
    matrix = _stabilize_covariance(
        covariance,
        dimension,
        floor,
        "Cholesky covariance",
    )
    jitter = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + jitter * np.eye(dimension))
        except np.linalg.LinAlgError:
            jitter = floor if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("covariance is not positive definite")


def _weighted_mean(
    points: np.ndarray,
    weights: np.ndarray,
    angle_indices: tuple[int, ...],
) -> np.ndarray:
    mean = weights @ points
    for index in angle_indices:
        mean[index] = np.arctan2(
            weights @ np.sin(points[:, index]),
            weights @ np.cos(points[:, index]),
        )
    return mean


def _weighted_covariance(
    points: np.ndarray,
    mean: np.ndarray,
    weights: np.ndarray,
    angle_indices: tuple[int, ...],
) -> np.ndarray:
    covariance = np.zeros((len(mean), len(mean)))
    for point, weight in zip(points, weights):
        difference = point - mean
        _wrap_selected(difference, angle_indices)
        covariance += weight * np.outer(difference, difference)
    return 0.5 * (covariance + covariance.T)


def _process_covariance(
    state_model: IStateModel,
    x: np.ndarray,
    dt: float,
    process_noise: float,
    dimension: int,
    floor: float,
) -> np.ndarray:
    return _stabilize_covariance(
        state_model.Q_discrete_white_noise(x, dt, process_noise),
        dimension,
        floor,
        "process covariance",
    )


def _validate_angle_indices(indices: tuple[int, ...], dimension: int) -> None:
    if any(index >= dimension for index in indices):
        raise ValueError("an angle index is outside the vector")


def _wrap_selected(vector: np.ndarray, indices: tuple[int, ...]) -> None:
    for index in indices:
        vector[index] = (vector[index] + np.pi) % (2.0 * np.pi) - np.pi
