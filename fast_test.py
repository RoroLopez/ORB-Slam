import numpy as np
np.set_printoptions(suppress=True)

import pandas as pd
import cv2 as cv
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
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
t = np.zeros(shape=(3,1))

F = 1
#F = 272.93867730682706

K = np.array([[F,0,W//2], [0,F,H//2], [0,0,1]])
Kinv = np.linalg.inv(K)

orb = cv.ORB_create(2500, scaleFactor=1.8, nlevels=12)
fast = cv.FastFeatureDetector_create()
bf = cv.BFMatcher(cv.NORM_HAMMING)


def features(frame):
    pts = cv.goodFeaturesToTrack(frame, 3000, qualityLevel=0.01, minDistance=7)
    key_pts = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    key_pts, descriptors = orb.compute(frame, key_pts)

    return key_pts, descriptors

def extractRt(F):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
    U,d,Vt = np.linalg.svd(F)
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U,W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:,2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

def normalize(inv, pts):
    return np.dot(inv, add_ones(pts).T).T[:,0:2]

def denormalize(K, pts):
    ret = np.dot(K, np.array([pts[0], pts[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def get_matches(Kinv,p1,p2):
    matches = bf.knnMatch(des1,des2,k=2)
    good = []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            pt1 = p1['kp1'][m.queryIdx].pt
            pt2 = p2['kp2'][m.trainIdx].pt
            good.append((pt1,pt2))
        
    good = np.array(good)

    good[:,0,:] = normalize(Kinv,good[:,0,:])
    good[:,1,:] = normalize(Kinv,good[:,1,:])

    model,inliers = ransac((good[:,0], good[:,1]),
                            FundamentalMatrixTransform,
                            min_samples=8, residual_threshold=0.5,
                            max_trials=100)

    good = good[inliers]

    pose = extractRt(model.params)

    return good, pose

while True:
    original_frame = cv.resize(frame, (W, H))
    resize_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)

    kp2, des2 = features(resize_frame)

    if len(kp1) == 0:
        kp1, des1, last_frame = kp2, des2, resize_frame
        ret, frame = cap.read()
        continue;
    else:
        img = cv.drawKeypoints(resize_frame,kp2,None,color=(255,0,0), flags=None)

    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    x1,x2 = [],[]

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            good.append((pt1, pt2))
        
            # if np.linalg.norm(np.diff([pt1,pt2])) < 0.5*np.linalg.norm([W,H]) and m.distance < 32:
            #     if m.queryIdx not in x1 and m.trainIdx not in x2:
            #         x1.append(m.queryIdx)
            #         x2.append(m.trainIdx)

    good = np.array(good)

    # print(len(x1))
    print(len(good))

    good[:,0,:] = normalize(Kinv, good[:,0,:])
    good[:,1,:] = normalize(Kinv, good[:,1,:])

    model, inliers = ransac((good[:,0], good[:, 1]),
                            FundamentalMatrixTransform,
                            min_samples=8, residual_threshold=0.5,
                            max_trials=100)

    good = good[inliers]

    pose = extractRt(model.params)
    # print(pose)
    # print(pose[:,3])
    # maybe = denormalize(K,pose[:,3])
    # print(maybe)

    for pt1, pt2 in good:
        u1,v1 = denormalize(K,pt1)
        u2,v2 = denormalize(K,pt2)

        # u1,v1 = map(lambda x: int(round(x)), pt1)
        # u2,v2 = map(lambda x: int(round(x)), pt2)
        img = cv.circle(original_frame, (u1,v1), color=(255,255,0), radius=3)
        cv.line(original_frame, (u1,v1), (u2,v2), color=(255,0,0))


    cv.imshow('frame', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    kp1, des1, last_frame = kp2, des2, resize_frame
    ret, frame = cap.read()

cap.release()
cv.destroyAllWindows()