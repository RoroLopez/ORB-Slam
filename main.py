import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('./video/testVideo.mp4')
ret, frame = cap.read()

height, width = frame.shape[:2]

W = int(width / 4)
H = int(height / 4)

orb = cv.ORB_create(1500, scaleFactor=1.8, nlevels=10)
fast = cv.FastFeatureDetector_create()
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher()

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


# def process_features(frame):
#     kp = cv.goodFeaturesToTrack(frame, 3000, 0.01, 10)
#     kp = np.int0(kp)
#     return kp

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

while True:
    resize_frame = cv.resize(frame, (W, H))
    resize_frame = cv.cvtColor(resize_frame, cv.COLOR_BGR2GRAY)

    # kp = process_features(resize_frame)
    # print(len(kp)

    # for p in kp:
    #     x,y = p.ravel()
    #     cv.circle(resize_frame, (x,y), 3, color=(0,0,255), thickness=-1)

    good = []
    pts1 = []
    pts2 = []

    kp2, des2 = process_features_orb(resize_frame)

    if len(kp1) == 0:
        kp1, des1, last_frame = kp2, des2, resize_frame
        continue;
    else:
        # original BF matcher with NORM_HAMMING
        # matches = bf.match(des1, des2)
        # matches = sorted(matches, key = lambda x:x.distance)

        # knnMatch matcher
        # matches = bf.knnMatch(des1, des2, k=2)
        # for m,n in matches:
        #     if m.distance < 0.75*n.distance:
        #         good.append([m])

        # FLANN based Matcher
        matches = flann.knnMatch(des1,des2,k=2)
        matchesMask = [[0,0] for i in range(len(matches))]
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                # matchesMask[i] = [1,0]
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

    if len(kp1) != 0:
        pts2 = np.float32(pts2)
        pts1 = np.float32(pts1)
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, 1, 0.99)

        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        # lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2),2,F)
        # lines1 = lines1.reshape(-1,3)
        # img5,img6 = drawlines(last_frame,resize_frame,lines1,pts1,pts2)

        # lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2),1,F)
        # lines2 = lines2.reshape(-1,3)
        # img3,img4 = drawlines(resize_frame,last_frame,lines2,pts2,pts1)

        # pt1 = np.array([[pts1[0][0]], [pts1[0][0]], [1]])
        # pt2 = np.array([[pts2[0][0], pts2[0][1], 1]])

        E, _ = cv.findEssentialMat(pts1, pts2, F, cv.RANSAC, 0.99, 1.0, None)

        _, R, t, _ = cv.recoverPose(E, pts1, pts2, F, R, t, None)

        # print(R)

        print(t)
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

        kp1, des1, last_frame = kp2, des2, resize_frame


        # drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor=None, singlePointColor=None, matchesMask=None, flags=None)
        img2 = cv.drawKeypoints(resize_frame,kp2,outImage=None,color=(0,255,0),flags=0)

    cv.imshow('frame', img2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()


cap.release()
cv.destroyAllWindows()

# pose calculation code
 
# pt1 = np.array([[pts1[0][0]], [pts1[0][1]], [1]])
# pt2 = np.array([[pts2[0][0], pts2[0][1], 1]])
 
# print pt1
# print pt2
 
# print
# print "The fundamental matrix is"
# print F
# print
 
# # Should be close to 0
# print "Fundamental matrix error check: %f"%np.dot(np.dot(pt2,F),pt1)
# print
 
# E = K.T.dot(F).dot(K)
 
# print "The essential matrix is"
# print E
# print
 
# U, S, Vt = np.linalg.svd(E)
# W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
# first_inliers = []
# second_inliers = []
# for i in range(len(pts1)):
#     first_inliers.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
#     second_inliers.append(K_inv.dot([pts2[i][0], pts2[i][1], 1.0]))
 
# # First choice: R = U * W * Vt, T = u_3
# R = U.dot(W).dot(Vt)
# T = U[:, 2]
 
# # Start degeneracy checks
# if not degeneracyCheckPass(first_inliers, second_inliers, R, T):
#     # Second choice: R = U * W * Vt, T = -u_3
#     T = - U[:, 2]
#     if not degeneracyCheckPass(first_inliers, second_inliers, R, T):
#         # Third choice: R = U * Wt * Vt, T = u_3
#         R = U.dot(W.T).dot(Vt)
#         T = U[:, 2]
#         if not degeneracyCheckPass(first_inliers, second_inliers, R, T):
#             # Fourth choice: R = U * Wt * Vt, T = -u_3
#             T = - U[:, 2]
 
# print "Translation matrix is"
# print T
# print "Modulus is %f" % np.sqrt((T[0]*T[0] + T[1]*T[1] + T[2]*T[2]))
# print "Rotation matrix is"
# print R
 
# # Decomposing rotation matrix
# pitch = np.arctan2(R[1][2], R[2][2]) * 180/3.1415
# yaw = np.arctan2(-R[2][0], np.sqrt(R[2][1]*R[2][1] + R[2][2]*R[2][2])) * 180/3.1415
# roll = np.arctan2(R[1][0],  R[0][0]) * 180/3.1415
 
# print "Roll: %f, Pitch: %f, Yaw: %f" %(roll , pitch , yaw)
 
# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()