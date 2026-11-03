import numpy as np
from scipy.linalg import block_diag
from tracking.models.state_model_base import IStateModel, permute


class NearlyConstantVelocity(IStateModel):
    def f(self, x: np.array, dt: float) -> np.array:
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        return F @ x

    def FJacobian(self, x: np.array, dt: float) -> np.ndarray:
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        return F

    def Q_continuous_white_noise(
        self,
        x: np.array,
        dt: float,
        process_noise: float
    ) -> np.ndarray:
        Q_1d = np.array([
            [(dt**3)/3.0, (dt**2)/2.0],
            [(dt**2)/2.0,  dt        ]
        ]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,2,1,3]
        return permute(Q,id_x)

    def Q_discrete_white_noise(
        self,
        x: np.array,
        dt: float,
        process_noise: float
    ) -> np.ndarray:
        Q_1d = np.array([
            [(dt**4)/4.0, (dt**3)/2.0],
            [(dt**3)/2.0,  dt**2     ]
        ]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,2,1,3]
        return permute(Q, id_x)