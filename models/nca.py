import numpy as np
from scipy.linalg import block_diag
from tracking.models.state_model_base import IStateModel, permute


class NearlyConstantAcceleration(IStateModel):
    def f(self, x: np.array, dt: float) -> np.array:
        acc = 0.5 * dt**2
        F = np.array([
            [1.0, 0.0, dt,  0.0, acc, 0.0],
            [0.0, 1.0, 0.0, dt,  0.0, acc],
            [0.0, 0.0, 1.0, 0.0, dt,  0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return F @ x

    def FJacobian(self, x: np.array, dt: float) -> np.ndarray:
        acc = 0.5 * dt**2
        F = np.array([[1.0, 0.0, dt,  0.0, acc, 0.0],
                      [0.0, 1.0, 0.0, dt,  0.0, acc],
                      [0.0, 0.0, 1.0, 0.0, dt,  0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return F

    def Q_continuous_white_noise(
        self,
        x: np.array,
        dt: float,
        process_noise: float
    ) -> np.ndarray:
        # 連続白色モデル
        Q_1d = np.array([[(dt**5)/20.0, (dt**4)/8.0, (dt**3)/6.0],
                         [(dt**4)/8.0,  (dt**3)/3.0, (dt**2)/2.0],
                         [(dt**3)/6.0,  (dt**2)/2.0,  dt        ]]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,3,1,4,2,5]
        return permute(Q, id_x)

    def Q_discrete_white_noise(
        self,
        x: np.array,
        dt: float,
        process_noise: float
    ) -> np.ndarray:
        # 区分白色モデル
        Q_1d = np.array([[(dt**4)/4.0, (dt**3)/2.0, (dt**2)/2.0],
                         [(dt**3)/2.0,  dt**2     ,  dt        ],
                         [(dt**2)/2.0,  dt        ,  1.0       ]]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,3,1,4,2,5]
        return permute(Q, id_x)