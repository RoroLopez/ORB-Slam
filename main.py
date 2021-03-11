import numpy as np
import cv2 as cv

import OpenGL.GL as gl
import pangolin

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
    # frame.kp, frame.des = get_features(frame.img)
    frames.append(frame)

    if len(frames) <= 1:
        ret, frame = cap.read()
        continue;

    idx1, idx2, Rt = get_pose(camera.Kinv, frames[-1], frames[-2])
    frames[-1].pose = np.dot(Rt, frames[-2].pose)
    # print(len(idx1))
    # print(frames[-1].pose)

    # homogenous 3D coordinates
    pts3d = triangulatePoints(frames[-1].pose, frames[-1].pose, frames[-1].kp[idx1], frames[-2].kp[idx2])
    pts3d /= pts3d[:, 3:]

    # reject points without enough "parallax"...
    # reject points behind the camera
    good_pts3d = (np.abs(pts3d[:,3]) > 0.005) & (pts3d[:, 2] > 0)
    pts3d = pts3d[good_pts3d]
    
    print(pts3d)

    for i,p in enumerate(pts3d):
        if not good_pts3d[i]:
            continue;
        pt = Point(p)
        pt.add_observation(frames[-1], idx1[i])
        pt.add_observation(frames[-2], idx2[i])

    for id1, id2 in zip(idx1, idx2):
        u1,v1 = denormalize(K,frames[-1].kp[id1])
        u2,v2 = denormalize(K,frames[-2].kp[id2])
        img = cv.circle(original_frame, (u1,v1), color=(255,255,0), radius=3)
        cv.line(original_frame, (u1,v1), (u2,v2), color=(255,0,0))

    cv.imshow('frame', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()


cap.release()
cv.destroyAllWindows()