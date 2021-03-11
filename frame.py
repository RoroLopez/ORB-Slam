import numpy as np
from matcher import get_features, normalize

IRt = np.eye(4)
class Frame:
    def __init__(self, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.w, self.h = img.shape[:2]
        self.img = img
        self.pose = IRt

        pts, self.des = get_features(img)
        self.kp = normalize(self.Kinv, pts)