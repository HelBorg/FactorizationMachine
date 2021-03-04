import numpy as np
from scipy import sparse


class Statistics:
    def __init__(self):
        pass

    def r2(self, real_value, predictions):
        error = predictions - real_value
        disp = real_value - real_value.mean()
        return 1 - np.sum(error ** 2) / np.sum(disp ** 2)

    def rmse(self, real_value, predictions):
        return np.sqrt(np.sum((real_value - predictions) ** 2) / len(predictions))

    def get_statistics(self, real_value, predictions):
        r2 = self.r2(real_value, predictions)
        rmse = self.rmse(real_value, predictions)
        return r2, rmse
