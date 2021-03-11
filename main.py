import numpy as np
import cv2 as cv

import OpenGL.GL as gl
import pangolin

from multiprocessing import Process, Queue
import time
import sys

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

# global map
frames = []
points = []

def triangulatePoints(pose1, pose2, pts1, pts2):
    return cv.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts1.T).T


q = Queue()
# viewer = Process(target=generate_map, args=(self.q,))

# gonna skip id identification of the frames and jump straight to the map generation in another thread..
def generate_map():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)

    darr = None
    while not pangolin.ShouldQuit():
        darr = q.get(darr is None)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)

        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(d[:3, 3] for d in darr[0])

        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(d for d in darr[1])

        pangolin.FinishFrame()

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []

    def display_map(self):
        for f in self.frames:
            print(f.id)
            print(f.pose)
    

mapp = Map()

class Point(object):
    def __init__(self, mapp, loc):
        self.location = loc
        self.frames = []
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)
    
    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)

while True:
    original_frame = cv.resize(frame, (W, H))
    resized_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)

    frame = Frame(resized_frame, mapp, K)
    frames.append(frame)

    if frame.id == 0:
        ret, frame = cap.read()
        continue;
    
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = get_pose(camera.Kinv, f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    # homogenous 3D coordinates
    pts3d = triangulatePoints(f1.pose, f2.pose, f1.kp[idx1], f2.kp[idx2])
    pts3d /= pts3d[:, 3:]

    # reject points without enough "parallax"...
    # reject points behind the camera
    good_pts3d = (np.abs(pts3d[:,3]) > 0.005) & (pts3d[:, 2] > 0)
    pts3d = pts3d[good_pts3d]
    
    # print(pts3d)

    for i,p in enumerate(pts3d):
        if not good_pts3d[i]:
            continue;
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for id1, id2 in zip(idx1, idx2):
        u1,v1 = denormalize(K,f1.kp[id1])
        u2,v2 = denormalize(K,f2.kp[id2])
        img = cv.circle(original_frame, (u1,v1), color=(255,255,0), radius=3)
        cv.line(original_frame, (u1,v1), (u2,v2), color=(255,0,0))

    cv.imshow('frame', img)
    mapp.display_map()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()


cap.release()
cv.destroyAllWindows()