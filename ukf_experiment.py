from tracking.filters.ukf import UnscentedKalmanFilter
from tracking.filters.srukf import SquareRootUnscentedKalmanFilter
from tracking.sigma_points import MerweScaledSigmaPoints, SquareRootMerweScaledSigmaPoints
from asrsim.radar_simulator import RadarSimulator
from target.target import Target
from target.target_state import TargetState
from target.manuever_model_base import NearlyConstantVelocityModel, NearlyConstantAccelerationModel, NearlyConstantTurnModel, NearlyConstantJerkModel
from tracking.filters.filter_state import FilterState
import tracking.models.state_model_base as IStateModel
import tracking.measurement_model_strategy as IMeasurementModel
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import qr


# レーダー初期化
pos = np.array([0, 0]).T
dt = 4.0
radar = RadarSimulator(pos, dt, range_std = 10.0, azimuth_std = math.radians(0.1))

# ターゲット初期化
# target_ini_state = TargetState(np.array([100.0,100.0,250.0,250.0,0.0,0.0,0.0,0.0,0.0]))
# target = Target(target_ini_state, NearlyConstantVelocityModel())
# target_ini_state = TargetState(np.array([100.0,100.0,50.0,50.0,10.0,10.0,0.0,0.0,0.0]))
# target = Target(target_ini_state, NearlyConstantAccelerationModel())
# target_ini_state = TargetState(np.array([130,120,90,110,10,10,3,3,0.0]))
# target = Target(target_ini_state, NearlyConstantJerkModel())
target_ini_state = TargetState(np.array([500.0, 500.0, 25.0, 25.0, 0, 0, 0, 0, -0.05]))
target = Target(target_ini_state, NearlyConstantTurnModel())

# カルマンフィルタ初期化

# x0 = np.array([100.0,100.0,250.0,250.0]).T
# P0 = diag([100.0**2,100.0**2,10.0**2,10.0**2])
# x0 = np.array([100.0,100.0,50.0,50.0,10.0,10.0]).T
# P0 = diag([100.0**2,100.0**2,50.0**2,50.0**2,3.0**2,3.0**2])
# x0 = np.array([130,130,25,25,3,3]).T
# P0 = diag([100,100,3,3,1,1])
x0 = np.array([500.0,500.0,25.0,25.0,-0.05]).T
P0 = np.diag([500.0**2,500.0**2,100.0**2,100.0**2,0.05**2])
# S0 = np.linalg.cholesky(P0)

state = FilterState(x0, P0)
# state = FilterState(x=x0, S=S0)
# UKFモデル設定
ncv_state_model = IStateModel.NearlyConstantVelocity()
ncv_measurement_model = IMeasurementModel.NearlyConstantVelocity(radar.pos)

nca_state_model = IStateModel.NearlyConstantAcceleration()
nca_measurement_model = IMeasurementModel.NearlyConstantAcceleration(radar.pos)

ncj_state_model = IStateModel.NearlyConstantJerk()
ncj_measurement_model = IMeasurementModel.NearlyConstantJerk(radar.pos)

nct_state_model = IStateModel.NearlyConstantTurn()
nct_measurement_model = IMeasurementModel.NearlyConstantTurn(radar.pos)

#sigma_points_model = MerweScaledSigmaPoints(alpha=0.01, beta=2.0, kappa=3.0-len(x0))
#sigma_points_model = MerweScaledSigmaPoints(alpha=0.05, beta=2.0, kappa=0.0)
sigma_points_model = MerweScaledSigmaPoints(alpha=0.01, beta=2.0, kappa=0.0)
sigma_points_model = SquareRootMerweScaledSigmaPoints(alpha=0.01, beta=2.0, kappa=0.0)

# ukf = UnscentedKalmanFilter(ncv_state_model, ncv_measurement_model, sigma_points_model)
#ukf = UnscentedKalmanFilter(nca_state_model, nca_measurement_model, sigma_points_model)
# ukf = UnscentedKalmanFilter(ncj_state_model, ncj_measurement_model, sigma_points_model)
ukf = UnscentedKalmanFilter(nct_state_model, nct_measurement_model, sigma_points_model)
#ukf = SquareRootUnscentedKalmanFilter(nct_state_model, nct_measurement_model, sigma_points_model)

# 推定結果格納用
result_p = []
result_f = []
ground_truth = []
ground_truth.append(target_ini_state)

result_p.append(state.x)

# シナリオ時間設定
scenario_time = np.arange(0, 100, dt)

# 観測雑音共分散行列
R = np.diag([radar.range_std**2, radar.azimuth_std**2])
R = np.linalg.cholesky(R)

# 実行時間計測開始
start_time = time.perf_counter()

# シナリオスタート
for epoch in scenario_time:
    if epoch != 0:
        # UKF時間更新
        state = ukf.time_update(state, dt)
        result_p.append(state.x)

    # 観測値生成
    obs = radar.observe(target)
    # UKF観測更新
    state, z, res = ukf.measurement_update(state, obs, R)
    result_f.append(state.x)

    target_state = target.update(dt, 0.0)
    ground_truth.append(target_state)

# 実行時間計測終了
elapsed_time = time.perf_counter() - start_time
print(f"Execute Time:{elapsed_time:.6f}sec.")

# 真値/推定結果プロット
xs = [t.pos[0] for t in ground_truth]
ys = [t.pos[1] for t in ground_truth]
plt.scatter(xs, ys, marker="v", s=5.0)
plt.scatter(np.array(result_f)[:, 0], np.array(result_f)[:, 1], marker="o", s=5.0, color="red")
#plt.scatter(np.array(result_p)[:,0], np.array(result_p)[:,1], marker="x",s=5.0)
plt.grid(True, which="both")
plt.axis("equal")
plt.show()

fig, axis = plt.subplots(2,1)
axis[0].set_title("Position")
axis[0].plot(scenario_time, np.array(xs[0:100]), label="Ground Truth")
axis[0].plot(scenario_time, np.array(result_f)[:,0], label="Filter")
axis[0].legend()
# axis[1].set_title("Velocity")
# axis[1].plot(scenario_time, [t.vel[0] for t in ground_truth], label="Ground Truth")
# axis[1].plot(scenario_time, [t.vel[0] for t in result_f], label="Filter")
# axis[1].plot(scenario_time, [t.vel[0] for t in result_p], label="Predict")
# axis[1].legend()
plt.show()



