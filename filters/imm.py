import numpy as np


class InteractingMultipleModelFilter:
    def __init__(self, filters, transition_matrix, mode_probabilities):
        self.filters = filters
        self.M = len(filters)

        self.PI = transition_matrix
        self.mu = mode_probabilities

        self.c_j = np.zeros(self.M)

    # 混合確率を計算する
    def calc_mixing_probabilities(self):
        # 正規化定数の計算
        self.c_j = self.PI.T @ self.mu

        # 混合確率の計算
        for i in range(self.M):
            if self.c_j[i] <= 0:
                raise ValueError()
            self.omega[:, i] = self.PI[:, i] * self.mu / self.c_j[i]

    # ミキシング
    def mixing(self):
        pass

    # 時間更新
    def time_update(self, dt):
        self.calc_mixing_probabilities()
        self.mixing()

        # フィルタ毎に時間更新
        for filter in self.filters:
            filter.time_update(dt)

    #　観測更新
    def measurement_updata(self, z):
        # フィルタ毎に観測更新
        for filter in self.filters:
            filter.measurement_update(z)

