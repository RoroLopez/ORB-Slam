class Frame:
    def __init__(self, img):
        self.w, self.h = img.shape[:2]
        self.img = img
        self.kp = None
        self.des = None
        self.pose = None