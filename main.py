import numpy as np
import pandas as pd
import cv2 as cv
<<<<<<< HEAD

from matcher import Frame, get_pose, get_features, denormalize
=======
from matplotlib import pyplot as plt
>>>>>>> parent of 5adb20e... refactor

cap = cv.VideoCapture('./video/testVideo.mp4')
ret, frame = cap.read()

height, width = frame.shape[:2]

W = int(width / 4)
H = int(height / 4)

<<<<<<< HEAD
frames = []

IRt = np.eye(3)
while True:
    original_frame = cv.resize(frame, (W, H))
    resized_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)
    
    frame = Frame(resized_frame, K)
    frames.append(frame)

    if len(frames) <= 1:
=======
orb = cv.ORB_create(2000, scaleFactor=1.8, nlevels=10)
fast = cv.FastFeatureDetector_create()
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher(cv.NORM_HAMMING)

# FLANN based Matcher
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

kp1 = []
des1 = []
last_frame = []

R = np.zeros(shape=(3,3))
t = np.zeros(shape=(3,3))

# nt 	nfeatures = 500,
# float 	scaleFactor = 1.2f,
# int 	nlevels = 8,
# int 	edgeThreshold = 31,
# int 	firstLevel = 0,
# int 	WTA_K = 2,
# ORB::ScoreType 	scoreType = ORB::HARRIS_SCORE,
# int 	patchSize = 31,
# int 	fastThreshold = 20 


def process_features(frame):
    kp = cv.goodFeaturesToTrack(frame, 3000, 0.01, 10)
    kp = np.int0(kp)
    return kp

def process_features_fast(frame):
    kp = fast.detect(frame, None)
    return kp

def process_features_orb(frame):
    kp, des = orb.detectAndCompute(frame, None)
    return kp, des

def degeneracyCheckPass(first_points, second_points, rot, trans):
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)
 
        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False
 
    return True

def generate_path(rotations, translations):
    path = []
    current_point = np.array([0,0,0])

    for R, t in zip(rotations, translations):
        path.append(current_point)
        # current_point = current_point + t.reshape((3,))
    
    return np.array(path)

id = 0
while True:
    resize_frame = cv.resize(frame, (W, H))
    resize_frame = cv.cvtColor(resize_frame, cv.COLOR_BGR2GRAY)

    # kp = process_features(resize_frame)

    ret = []
    pts1, pts2 = [], []

    kp2, des2 = process_features_orb(resize_frame)
    # print(len(kp2))
    # kp2 = process_features_fast(frame)

    if len(kp1) == 0:
        kp1, des1, last_frame = kp2, des2, resize_frame
>>>>>>> parent of 5adb20e... refactor
        ret, frame = cap.read()
        continue;
    else:
        # original BF matcher with NORM_HAMMING
        # matches = bf.match(des1, des2)
        # matches = sorted(matches, key = lambda x:x.distance)

<<<<<<< HEAD
    idx1, idx2, Rt, pts = get_pose(frames[-1], frames[-2])

    pts3D = cv.triangulatePoints(frames[-1].pose[:3], Rt[:-1, :], pts[:, 0].T, pts[:,1].T).T
    # print(pts3D)
    frames[-1].pose = np.dot(Rt, frames[-2].pose)
    print(frames[-1].pose)

    # pts3D = cv.triangulatePoints(frames[-1].pose[:3], frames[-2].pose[:3], frames[-1].kp[idx1].T, frames[-2].kp[idx2].T).T
    # pts3D /= pts3D[:,3:]
    # good_pts3D = (np.abs(pts3D[:, 3]) > 0.005) & (pts3D[:,2] > 0)
    # print(good_pts3D)

    for pt1, pt2 in zip(frames[-1].kp[idx1], frames[-2].kp[idx2]):
        u1,v1 = denormalize(K,pt1)
        u2,v2 = denormalize(K,pt2)
=======
        # knnMatch matcher
        matches = bf.knnMatch(des1, des2, k=2)
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt

        
        print(len(pt1))

        # FLANN based Matcher
        # matches = flann.knnMatch(des1,des2,k=2)
        # matchesMask = [[0,0] for i in range(len(matches))]
        # for i,(m,n) in enumerate(matches):
        #     if m.distance < 0.7*n.distance:
        #         # matchesMask[i] = [1,0]
        #         pts2.append(kp2[m.trainIdx].pt)
        #         pts1.append(kp1[m.queryIdx].pt)
>>>>>>> parent of 5adb20e... refactor

        # draw_params = dict(matchColor = (0,255,0),
        #            singlePointColor = (255,0,0),
        #            matchesMask = matchesMask,
        #            flags = 0)

    if len(kp1) != 0:
        pts2 = np.float32(pts2)
        pts1 = np.float32(pts1)
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 1, 0.99)

        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        print(len(pts1) == len(pts2))

        if id == 0:
            E, _ = cv.findEssentialMat(pts1, pts2, F, cv.RANSAC, 0.99, 1.0, None)
            _, R, t, _ = cv.recoverPose(E, pts1, pts2, F, R, t, None)
            id += 1
        else:
            E, _ = cv.findEssentialMat(pts1, pts2, F, cv.RANSAC, 0.99, 1.0, None)
            points, R, t, mask = cv.recoverPose(E, pts1, pts2, F, R.copy(), t.copy(), None)

        # Decomposing rotation matrix
        # pitch = np.arctan2(R[1][2], R[2][2]) * 180/3.1415
        # yaw = np.arctan2(-R[2][0], np.sqrt(R[2][1]*R[2][1] + R[2][2]*R[2][2])) * 180/3.1415
        # roll = np.arctan2(R[1][0],  R[0][0]) * 180/3.1415
        
        # print("Roll: {0}\t, Pitch: {1}\t, Yaw: {2}".format(roll,pitch,yaw))

        
        # original image for matches
        # img2 = cv.drawMatches(last_frame,kp1,resize_frame,kp2,matches[:1000], outImg=None, flags=2)

        # knnMatch matcher
        # img2 = cv.drawMatchesKnn(last_frame, kp1, resize_frame, kp2, good, outImg=None, flags=2)

        # FLANN based Matcher
        # for m in matches:
        #     print("distance: {0}".format(m[0].distance))
        # img2 = cv.drawMatchesKnn(last_frame,kp1,resize_frame,kp2,matches,None,**draw_params)

        img2 = cv.drawKeypoints(resize_frame,kp2,outImage=None,color=(0,255,0),flags=0)

        kp1, des1, last_frame = kp2, des2, resize_frame


        # drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor=None, singlePointColor=None, matchesMask=None, flags=None)

    cv.imshow('frame', img2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()


cap.release()
cv.destroyAllWindows()