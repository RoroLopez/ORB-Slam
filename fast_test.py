import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('./video/testVideo.mp4')
ret, frame = cap.read()

height, width = frame.shape[:2]

W = int(width / 4)
H = int(height / 4)

kp1 = []
des1 = []
last_frame = []

R = np.zeros(shape=(3,3))
t = np.zeros(shape=(3,3))

orb = cv.ORB_create(2500, scaleFactor=1.8, nlevels=12)
fast = cv.FastFeatureDetector_create()
bf = cv.BFMatcher(cv.NORM_HAMMING)


def features(frame):
    pts = cv.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
    key_pts = [cv.KeyPoint(x=f[0][0], y=[0][1], _size=20) for f in pts]
    key_pts, descriptors = orb.compute(frame, key_pts)

    return key_pts, descriptors

def features_orb(frame):
    kps, des = orb.detectAndCompute(frame, None)

    return kps, des


def normalize(count_inv, pts):
    return np.dot(count_inv, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T[:, 0:2]

def denormalize(count, pt):
    ret = np.dot(count, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]

    return int(round(ret[0])), int(round(ret[1]))

while True:
    resize_frame = cv.resize(frame, (W, H))
    resize_frame = cv.cvtColor(resize_frame, cv.COLOR_BGR2GRAY)

    kp2, des2 = features_orb(resize_frame)

    if len(kp1) == 0:
        kp1, des1, last_frame = kp2, des2, resize_frame
        ret, frame = cap.read()
        continue;
    else:
        img = cv.drawKeypoints(resize_frame,kp2,None,color=(255,0,0), flags=None)

    matches = bf.knnMatch(des1, des2, k=2)

    # img = cv.drawMatches(last_frame,kp1,resize_frame,kp2,matches[:1000],outImg=None,matchColor=(255,0,0),singlePointColor=None,matchesMask=None,flags=2)
    
    good = []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append((kp1[m.queryIdx], kp2[m.trainIdx]))
    
    for pt1, pt2 in good:
        u1,v1 = map(lambda x: int(round(x)), pt1.pt)
        u2,v2 = map(lambda x: int(round(x)), pt2.pt)
        img = cv.circle(resize_frame, (u1,v1), color=(255,255,0), radius=3)
    cv.imshow('frame', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    kp1, des1, last_frame = kp2, des2, resize_frame
    ret, frame = cap.read()

cap.release()
cv.destroyAllWindows()