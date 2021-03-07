import numpy as np
import pandas as pd
import cv2 as cv

from matcher import Camera, get_pose, get_features, denormalize
from frame import Frame, IRt

cap = cv.VideoCapture('./video/testVideo.mp4')
ret, frame = cap.read()

height, width = frame.shape[:2]

W = int(width / 4)
H = int(height / 4)
F = 272.93867730682706
K = np.array([[F,0,W//2], [0,F,H//2], [0,0,1]])

camera = Camera(K)

frames = []

def triangulatePoints(pose1, pose2, pts1, pts2):
    return cv.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts1.T).T


class Point(object):
    def __init__(self, loc):
        self.location = loc
        self.frames = []
        self.idxs = []
    
    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

while True:
    original_frame = cv.resize(frame, (W, H))
    resized_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)

    frame = Frame(resized_frame, K)
    frame.kp, frame.des = get_features(frame.img)
    frames.append(frame)

    if len(frames) <= 1:
        ret, frame = cap.read()
        continue;

    pts, Rt = get_pose(camera.Kinv, frames[-2], frames[-1])
    frames[-1].pose = np.dot(Rt, frames[-2].pose)
    # print(frames[-1].pose)

    # homogenous 3D coordinates
    pts3d = triangulatePoints(frames[-1].pose, frames[-1].pose, pts[:,0], pts[:,1])
    pts3d /= pts3d[:, 3:]

    # reject points without enough "parallax"...
    # reject points behind the camera
    good_pts3d = (np.abs(pts3d[:,3]) > 0.005) & (pts3d[:, 2] > 0)
    pts3d = pts3d[good_pts3d]
    
    print(pts3d)

    for pt1, pt2 in pts:
        u1,v1 = denormalize(K,pt1)
        u2,v2 = denormalize(K,pt2)
        img = cv.circle(original_frame, (u1,v1), color=(255,255,0), radius=3)
        cv.line(original_frame, (u1,v1), (u2,v2), color=(255,0,0))

    cv.imshow('frame', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()


cap.release()
cv.destroyAllWindows()