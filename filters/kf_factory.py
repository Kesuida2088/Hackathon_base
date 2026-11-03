from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import Any, TypeVar

import numpy as np

from tracking.filters.ckf import CubatureKalmanFilter
from tracking.filters.ekf import EKFSetting, ExtendedKalmanFilter
from tracking.filters.kf_base import FilterType, IKalmanFilter
from tracking.filters.srekf import (
    EKFSetting as SREKFSetting,
    SquareRootExtendedKalmanFilter,
)
from tracking.filters.srukf import SquareRootUnscentedKalmanFilter
from tracking.filters.ukf import UnscentedKalmanFilter
from tracking.measurement_model_strategy import (
    MeasurementModelStrategy,
    NearlyConstantAcceleration as NCAMeasurementModel,
    NearlyConstantJerk as NCJMeasurementModel,
    NearlyConstantTurn as NCTMeasurementModel,
    NearlyConstantVelocity as NCVMeasurementModel,
)
from tracking.models.state_model_base import (
    IStateModel,
    NearlyConstantAcceleration,
    NearlyConstantJerk,
    NearlyConstantTurn,
    NearlyConstantVelocity,
)
from tracking.sigma_points import (
    CubatureSigmaPoints,
    MerweScaledSigmaPoints,
    SigmaPointsStrategy,
    SquareRootMerweScaledSigmaPoints,
)


TEnum = TypeVar("TEnum", bound=Enum)


class StateModelType(Enum):
    NCV = auto()
    NCA = auto()
    NCJ = auto()
    NCT = auto()


class MeasurementModelType(Enum):
    NCV = auto()
    NCA = auto()
    NCJ = auto()
    NCT = auto()


class SigmaPointType(Enum):
    MERWE = auto()
    SQUARE_ROOT_MERWE = auto()
    CUBATURE = auto()


class KalmanFilterFactory:
    """Factory for composing filters with state and measurement models."""

    _STATE_MODEL_BUILDERS: dict[StateModelType, Callable[[], IStateModel]] = {
        StateModelType.NCV: NearlyConstantVelocity,
        StateModelType.NCA: NearlyConstantAcceleration,
        StateModelType.NCJ: NearlyConstantJerk,
        StateModelType.NCT: NearlyConstantTurn,
    }

    _MEASUREMENT_MODEL_BUILDERS: dict[
        MeasurementModelType,
        Callable[[np.ndarray], MeasurementModelStrategy],
    ] = {
        MeasurementModelType.NCV: NCVMeasurementModel,
        MeasurementModelType.NCA: NCAMeasurementModel,
        MeasurementModelType.NCJ: NCJMeasurementModel,
        MeasurementModelType.NCT: NCTMeasurementModel,
    }

    @classmethod
    def create(
        cls,
        filter_type: FilterType | str,
        state_model_type: StateModelType | str,
        measurement_model_type: MeasurementModelType | str,
        *,
        radar_pos: np.ndarray,
        ekf_setting: EKFSetting | SREKFSetting | None = None,
        sigma_point_type: SigmaPointType | str | None = None,
        sigma_points: SigmaPointsStrategy | None = None,
        sigma_point_kwargs: dict[str, Any] | None = None,
    ) -> IKalmanFilter:
        state_model = cls.create_state_model(state_model_type)
        measurement_model = cls.create_measurement_model(
            measurement_model_type,
            radar_pos=radar_pos,
        )
        return cls.create_filter(
            filter_type,
            state_model=state_model,
            measurement_model=measurement_model,
            ekf_setting=ekf_setting,
            sigma_point_type=sigma_point_type,
            sigma_points=sigma_points,
            sigma_point_kwargs=sigma_point_kwargs,
        )

    @classmethod
    def create_filter(
        cls,
        filter_type: FilterType | str,
        *,
        state_model: IStateModel,
        measurement_model: MeasurementModelStrategy,
        ekf_setting: EKFSetting | SREKFSetting | None = None,
        sigma_point_type: SigmaPointType | str | None = None,
        sigma_points: SigmaPointsStrategy | None = None,
        sigma_point_kwargs: dict[str, Any] | None = None,
    ) -> IKalmanFilter:
        filter_type = _coerce_enum(FilterType, filter_type)

        if filter_type == FilterType.EKF:
            setting = ekf_setting or EKFSetting(ite_max=1, ite_eps=1e-4)
            return ExtendedKalmanFilter(
                state_model=state_model,
                measurement_model=measurement_model,
                setting=setting,
            )

        if filter_type == FilterType.SREKF:
            setting = ekf_setting or SREKFSetting(ite_max=1, ite_eps=1e-4)
            return SquareRootExtendedKalmanFilter(
                state_model=state_model,
                measurement_model=measurement_model,
                setting=setting,
            )

        if filter_type == FilterType.UKF:
            sigma_points = sigma_points or cls.create_sigma_points(
                sigma_point_type or SigmaPointType.MERWE,
                **(sigma_point_kwargs or {}),
            )
            return UnscentedKalmanFilter(
                state_model=state_model,
                measurement_model=measurement_model,
                sigma_points_strategy=sigma_points,
            )

        if filter_type == FilterType.SRUKF:
            sigma_points = sigma_points or cls.create_sigma_points(
                sigma_point_type or SigmaPointType.SQUARE_ROOT_MERWE,
                **(sigma_point_kwargs or {}),
            )
            return SquareRootUnscentedKalmanFilter(
                state_model=state_model,
                measurement_model=measurement_model,
                sigma_points_strategy=sigma_points,
            )

        if filter_type == FilterType.CKF:
            return CubatureKalmanFilter(
                state_model=state_model,
                measurement_model=measurement_model,
            )

        raise ValueError(f"{filter_type.name} is not implemented")

    @classmethod
    def create_state_model(
        cls,
        model_type: StateModelType | str,
    ) -> IStateModel:
        model_type = _coerce_enum(StateModelType, model_type)
        return cls._STATE_MODEL_BUILDERS[model_type]()

    @classmethod
    def create_measurement_model(
        cls,
        model_type: MeasurementModelType | str,
        *,
        radar_pos: np.ndarray,
    ) -> MeasurementModelStrategy:
        model_type = _coerce_enum(MeasurementModelType, model_type)
        return cls._MEASUREMENT_MODEL_BUILDERS[model_type](radar_pos)

    @staticmethod
    def create_sigma_points(
        sigma_point_type: SigmaPointType | str,
        *,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> SigmaPointsStrategy:
        sigma_point_type = _coerce_enum(SigmaPointType, sigma_point_type)

        if sigma_point_type == SigmaPointType.MERWE:
            return MerweScaledSigmaPoints(
                alpha=alpha,
                beta=beta,
                kappa=kappa,
            )

        if sigma_point_type == SigmaPointType.SQUARE_ROOT_MERWE:
            return SquareRootMerweScaledSigmaPoints(
                alpha=alpha,
                beta=beta,
                kappa=kappa,
            )

        if sigma_point_type == SigmaPointType.CUBATURE:
            return CubatureSigmaPoints()

        raise ValueError(f"Unsupported sigma point type: {sigma_point_type}")

    @classmethod
    def register_state_model(
        cls,
        model_type: StateModelType,
        builder: Callable[[], IStateModel],
    ) -> None:
        cls._STATE_MODEL_BUILDERS[model_type] = builder

    @classmethod
    def register_measurement_model(
        cls,
        model_type: MeasurementModelType,
        builder: Callable[[np.ndarray], MeasurementModelStrategy],
    ) -> None:
        cls._MEASUREMENT_MODEL_BUILDERS[model_type] = builder


def _coerce_enum(enum_type: type[TEnum], value: TEnum | str) -> TEnum:
    if isinstance(value, enum_type):
        return value

    if isinstance(value, str):
        normalized = value.strip().upper().replace("-", "").replace("_", "")
        for member in enum_type:
            member_name = member.name.replace("_", "")
            if normalized == member_name:
                return member

    raise ValueError(f"Unsupported {enum_type.__name__}: {value}")
