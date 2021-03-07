import numpy as np
np.set_printoptions(suppress=True)
import cv2 as cv

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

def get_features(frame):
    orb = cv.ORB_create(2500, scaleFactor=1.8, nlevels=12)
    pts = cv.goodFeaturesToTrack(frame, 3000, qualityLevel=0.01, minDistance=7)
    key_pts = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    key_pts, descriptors = orb.compute(frame, key_pts)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in key_pts]), descriptors

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

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
    # print(ret)
    return ret

def get_pose(frame1, frame2):
    bf = cv.BFMatcher(cv.NORM_HAMMING)

    Kinv = frame1.Kinv

    matches = bf.knnMatch(frame1.des, frame2.des, k=2)
    good = []
    x1,x2 = [],[]

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            x1.append(m.queryIdx)
            x2.append(m.trainIdx)

            pt1 = frame1.kp[m.queryIdx]
            pt2 = frame2.kp[m.trainIdx]
            good.append((pt1,pt2))
        
    good = np.array(good)
    x1 = np.array(x1)
    x2 = np.array(x2)

    good[:,0,:] = normalize(Kinv,good[:,0,:])
    good[:,1,:] = normalize(Kinv,good[:,1,:])

    model, inliers = ransac((good[:,0], good[:,1]),
                            #FundamentalMatrixTransform,
                            EssentialMatrixTransform,
                            min_samples=8,
                            #residual_threshold=0.5,
                            residual_threshold=0.005,
                            max_trials=100)

    good = good[inliers]
    pose = extractRt(model.params)

    # return good, pose
    return x1[inliers], x2[inliers], pose, good


def normalize(Kinv, pts):
        return np.dot(Kinv, add_ones(pts).T).T[:,0:2]
    
def denormalize(K, pts):
    result = np.dot(K, np.array([pts[0], pts[1], 1.0]))
    result /= result[2]
    return int(round(result[0])), int(round(result[1]))

# initial pose
IRt = np.eye(4)

class Frame(object):
    def __init__(self, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.pose = IRt
        
        pts, self.des = get_features(img)
        self.kp = normalize(self.Kinv, pts)