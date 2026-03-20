import numpy as np

class FeatureNormalizer:
    def __init__(self):
        self.stats = {}

    def fit(self, feature_list):
        """
        feature_list: list of dicts
        """
        for key in feature_list[0]:
            vals = np.array([f[key] for f in feature_list])
            self.stats[key] = {
                "min": float(vals.min()),
                "max": float(vals.max())
            }

    def normalize(self, features):
        norm = {}
        for k, v in features.items():
            mn = self.stats[k]["min"]
            mx = self.stats[k]["max"]
            norm[k] = (v - mn) / (mx - mn + 1e-6)
            norm[k] = float(np.clip(norm[k], 0.0, 1.0))
        return norm
