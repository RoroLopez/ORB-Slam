import numpy as np

IRt = np.eye(4)
class Frame:
    def __init__(self, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.w, self.h = img.shape[:2]
        self.img = img
        self.kp = None
        self.des = None
        self.pose = IRt