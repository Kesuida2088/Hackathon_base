import numpy as np
from numpy import sin, cos, zeros, eye
from scipy.linalg import block_diag
from tracking.models.state_model_base import IStateModel, permute


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
            F[0, 2] = dt
            F[0, 4] = -0.5 * dt**2 * vy
            F[1, 3] = dt
            F[1, 4] = 0.5 * dt**2 * vx
            F[2, 4] = -dt * vy
            F[3, 4] = dt * vx
            return F

        omg2 = omg**2
        wt = omg * dt
        sin_wt = sin(wt)
        cos_wt = cos(wt)

        # ヤコビ行列F生成
        F = eye(len(x))
        F[0, 2] = sin_wt / omg
        F[0, 3] = -(1.0 - cos_wt) / omg
        F[0, 4] = ((wt * cos_wt - sin_wt) / omg2) * vx - ((wt * sin_wt - (1.0 - cos_wt)) / omg2) * vy
        F[1, 2] = (1.0 - cos_wt) / omg
        F[1, 3] = sin_wt / omg
        F[1, 4] = ((wt * sin_wt - (1.0 - cos_wt)) / omg2) * vx + ((wt * cos_wt - sin_wt) / omg2) * vy
        F[2, 2] = cos_wt
        F[2, 3] = -sin_wt
        F[2, 4] = -dt * sin_wt * vx - dt * cos_wt * vy
        F[3, 2] = sin_wt
        F[3, 3] = cos_wt
        F[3, 4] = dt * cos_wt * vx - dt * sin_wt * vy

        return F

    def Q_continuous_white_noise(
        self,
        x: np.array,
        dt: float,
        process_noise: float
    ) -> np.ndarray:
        v2 = x[2]**2 + x[3]**2
        q_pos = (v2 * dt**5) / 20.0
        q_vel = (v2 * dt**3) / 3.0
        Q = np.array([q_pos, q_pos, q_vel, q_vel, dt]) * process_noise
        return np.diag(Q)

    def Q_discrete_white_noise(
        self,
        x: np.array,
        dt: float,
        process_noise: float
    ) -> np.ndarray:
        Q_1d = np.array([
            [(dt**4)/4.0, (dt**3)/2.0],
            [(dt**3)/2.0,  dt**2     ]]
        ) * process_noise
        Q = block_diag(Q_1d, Q_1d, 0.02)
        id_x = [0,2,1,3,4]
        return permute(Q, id_x)
