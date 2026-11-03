import numpy as np
from numpy import sin, cos, zeros, eye
from scipy.linalg import block_diag
from abc import ABC, abstractmethod

def permute(mat: np.ndarray, perm: np.array):
    return mat[np.ix_(perm,perm)]


class IStateModel(ABC):
    @abstractmethod
    def f(self, x: np.array, dt: float):
        pass


    @abstractmethod
    def FJacobian(self, x: np.array, dt: float):
        pass


    @abstractmethod
    def Q_continuous_white_noise(self, x: np.array, dt: float, process_noise):
        pass


    @abstractmethod
    def Q_discrete_white_noise(self, x: np.array, dt: float, process_noise):
        pass


class NearlyConstantVelocity(IStateModel):
    def f(self, x: np.array, dt: float) -> np.array:
        F = np.array([[1.0, 0.0, dt,  0.0],
                      [0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        return F @ x


    def FJacobian(self, x: np.array, dt: float) -> np.ndarray:
        F = np.array([[1.0, 0.0, dt,  0.0],
                      [0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        return F


    def Q_continuous_white_noise(self, x: np.array, dt: float, process_noise: float) -> np.ndarray:
        Q_1d = np.array([[(dt**3)/3.0, (dt**2)/2.0],
                         [(dt**2)/2.0,  dt        ]]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,2,1,3]
        return permute(Q,id_x)


    def Q_discrete_white_noise(self, x: np.array, dt: float, process_noise: float) -> np.ndarray:
        Q_1d = np.array([[(dt**4)/4.0, (dt**3)/2.0],
                         [(dt**3)/2.0,  dt**2     ]]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,2,1,3]
        return permute(Q, id_x)


class NearlyConstantAcceleration(IStateModel):
    def f(self, x: np.array, dt: float) -> np.array:
        acc = 0.5 * dt**2
        F = np.array([[1.0, 0.0, dt,  0.0, acc, 0.0],
                      [0.0, 1.0, 0.0, dt,  0.0, acc],
                      [0.0, 0.0, 1.0, 0.0, dt,  0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return F @ x


    def FJacobian(self, x:np.array, dt: float) -> np.ndarray:
        acc = 0.5 * dt**2
        F = np.array([[1.0, 0.0, dt,  0.0, acc, 0.0],
                      [0.0, 1.0, 0.0, dt,  0.0, acc],
                      [0.0, 0.0, 1.0, 0.0, dt,  0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, dt ],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return F


    def Q_continuous_white_noise(self, x: np.array, dt: float, process_noise: float) -> np.ndarray:
        # 連続白色モデル
        Q_1d = np.array([[(dt**5)/20.0, (dt**4)/8.0, (dt**3)/6.0],
                         [(dt**4)/8.0,  (dt**3)/3.0, (dt**2)/2.0],
                         [(dt**3)/6.0,  (dt**2)/2.0,  dt        ]]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,3,1,4,2,5]
        return permute(Q, id_x)


    def Q_discrete_white_noise(self, x: np.array, dt: float, process_noise: float) -> np.ndarray:
        # 区分白色モデル
        Q_1d = np.array([[(dt**4)/4.0, (dt**3)/2.0, (dt**2)/2.0],
                         [(dt**3)/2.0,  dt**2     ,  dt        ],
                         [(dt**2)/2.0,  dt        ,  1.0       ]]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,3,1,4,2,5]
        return permute(Q, id_x)

class NearlyConstantJerk(IStateModel):
    def f(self, x: np.array, dt: float) -> np.array:
        acc = 1 / 2 * dt**2
        jerk1 = 1 / 6 * dt**3
        jerk2 = 1 / 2 * dt**2
        F = np.array([[1.0, 0.0, dt,  0.0, acc, 0.0, jerk1, 0.0 ],
                      [0.0, 1.0, 0.0, dt,  0.0, acc, 0.0,   jerk1],
                      [0.0, 0.0, 1.0, 0.0, dt,  0.0, jerk2, 0.0 ],
                      [0.0, 0.0, 0.0, 1.0, 0.0, dt,  0.0,   jerk2],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, acc,   0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,   acc],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   1.0]])
        return F @ x


    def FJacobian(self, x: np.array, dt: float) -> np.ndarray:
        acc = 1 / 2 * dt**2
        jerk1 = 1 / 6 * dt**3
        jerk2 = 1 / 2 * dt**2
        F = np.array([[1.0, 0.0, dt,  0.0, acc, 0.0, jerk1, 0.0 ],
                      [0.0, 1.0, 0.0, dt,  0.0, acc, 0.0,   jerk1],
                      [0.0, 0.0, 1.0, 0.0, dt,  0.0, jerk2, 0.0 ],
                      [0.0, 0.0, 0.0, 1.0, 0.0, dt,  0.0,   jerk2],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, acc,   0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,   acc],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,   0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   1.0]])
        return F


    def Q_continuous_white_noise(self, x: np.array, dt: float, process_noise: float) -> np.ndarray:
        # 連続白色モデル
        Q_1d = np.array([[(dt**5)/20.0, (dt**4)/8.0, (dt**3)/6.0],
                         [(dt**4)/8.0,  (dt**3)/3.0, (dt**2)/2.0],
                         [(dt**3)/6.0,  (dt**2)/2.0,  dt        ]]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,3,1,4,2,5]
        return permute(Q, id_x)


    def Q_discrete_white_noise(self, x: np.array, dt: float, process_noise: float) -> np.ndarray:
        # 区分白色モデル
        Q_1d = np.array([[(dt**4)/4.0, (dt**3)/2.0, (dt**2)/2.0],
                         [(dt**3)/2.0,  dt**2     ,  dt        ],
                         [(dt**2)/2.0,  dt        ,  1.0       ]]) * process_noise
        Q = block_diag(Q_1d, Q_1d)
        id_x = [0,3,1,4,2,5]
        return permute(Q, id_x)


class NearlyConstantTurn(IStateModel):
    _OMEGA_EPS = 1e-8

    def f(self, x: np.array, dt: float) -> np.array:
        px, py, vx, vy, omg = x

        if abs(omg) < self._OMEGA_EPS:
            return np.array([px + vx * dt, py + vy * dt, vx, vy, omg])

        wt = omg * dt
        sin_wt = sin(wt)
        cos_wt = cos(wt)

        x = zeros(len(x))
        x[0] = px + (sin_wt / omg) * vx - ((1.0 - cos_wt) / omg) * vy
        x[1] = py + ((1.0 - cos_wt) / omg) * vx + (sin_wt / omg) * vy
        x[2] = cos_wt * vx - sin_wt * vy
        x[3] = sin_wt * vx + cos_wt * vy
        x[4] = omg

        return x


    def FJacobian(self, x: np.array, dt: float) -> np.ndarray:
        _, _, vx, vy, omg = x

        if abs(omg) < self._OMEGA_EPS:
            F = eye(len(x))
            F[0,2] = dt
            F[0,4] = -0.5 * dt**2 * vy
            F[1,3] = dt
            F[1,4] = 0.5 * dt**2 * vx
            F[2,4] = -dt * vy
            F[3,4] = dt * vx
            return F

        omg2 = omg**2
        wt = omg * dt
        sin_wt = sin(wt)
        cos_wt = cos(wt)

        # ヤコビ行列F生成
        F = eye(len(x))
        F[0,2] = sin_wt / omg
        F[0,3] = -(1.0 - cos_wt) / omg
        F[0,4] = ((wt * cos_wt - sin_wt) / omg2) * vx - ((wt * sin_wt - (1.0 - cos_wt)) / omg2) * vy
        F[1,2] = (1.0 - cos_wt) / omg
        F[1,3] = sin_wt / omg
        F[1,4] = ((wt * sin_wt - (1.0 - cos_wt)) / omg2) * vx + ((wt * cos_wt - sin_wt) / omg2) * vy
        F[2,2] = cos_wt
        F[2,3] = -sin_wt
        F[2,4] = -dt * sin_wt * vx - dt * cos_wt * vy
        F[3,2] = sin_wt
        F[3,3] = cos_wt
        F[3,4] = dt * cos_wt * vx - dt * sin_wt * vy

        return F


    def Q_continuous_white_noise(self, x: np.array, dt: float, process_noise: float) -> np.ndarray:
        v2 = x[2]**2 + x[3]**2
        q_pos = (v2 * dt**5 )/ 20.0
        q_vel = (v2 * dt**3 )/ 3.0
        Q = np.array([q_pos, q_pos, q_vel, q_vel, dt]) * process_noise
        return np.diag(Q)


    def Q_discrete_white_noise(self, x: np.array, dt: float, process_noise: float) -> np.ndarray:
        Q_1d = np.array([[(dt**4)/4.0, (dt**3)/2.0],
                         [(dt**3)/2.0,  dt**2     ]]) * process_noise**2
        Q = block_diag(Q_1d, Q_1d, np.array([dt**2 * process_noise**2]))
        id_x = [0,2,1,3,4]
        return permute(Q, id_x)


class VelocitySinger(IStateModel):
    def f(self, dt):
        pass


    def FJacobian(self, dt):
        pass


    def Q_continuous_white_noise(self, dt):
        pass


    def Q_discrete_white_noise(self, dt, process_noise):
        pass


class AccelerationSinger(IStateModel):
    def f(self, dt):
        pass


    def FJacobian(self, dt):
        pass


    def Q_continuous_white_noise(self, dt):
        pass


    def Q_discrete_white_noise(self, dt, process_noise):
        pass

class JerkSinger(IStateModel):
    def f(self, dt):
        pass


    def FJacobian(self, dt):
        pass


    def Q_continuous_white_noise(self, dt):
        pass


    def Q_discrete_white_noise(self, dt, process_noise):
        pass
