from tracking.filters.kf_base import IKalmanFilter


class IMM:
    def __init__(self, filters: list[IKalmanFilter]):
        self.filters = filters

    def predict(self):
        pass

    def update(self):
        pass

    def mixing(self):
        pass

