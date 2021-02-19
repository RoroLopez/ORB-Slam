import numpy as np
import cv2 as cv

cap = cv.VideoCapture('./video/testVideo.mp4')
ret,frame = cap.read()

height, width = frame.shape[:2]

W = int(width / 4)
H = int(height / 4)

fast = cv.FastFeatureDetector_create(threshold=120, nonmaxSuppression=True)

R = np.zeros(shape=(3,3))
t = np.zeros(shape=(3,3))

lk_params = dict(winSize = (21,21),
                    maxLevel = 3,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1, 0.03))

feature_params = dict(maxCorners = 1500,
                        qualityLevel = 0.01,
                        minDistance = 7,
                        blockSize = 7)

kp1 = []
last_frame = []

# random colors
color = np.random.randint(0,255,(100,3))

p1 = []
st = []
err = []

def process_good_features(frame):
    return cv.goodFeaturesToTrack(frame, mask=None, **feature_params)

def process_features(frame):
    kp = fast.detect(frame, None)
    return np.array([x.pt for x in kp], dtype=np.float32).reshape(-1,1,2)

while True:
    resized_frame = cv.resize(frame, (W, H))
    resized_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(resized_frame)

    kp2 = process_features(resized_frame)
    # print(len(kp2))
    # kp2 = process_good_features(resized_frame)

    # numpy_array = []
    # outer = []
    # individual = []

    # for p in kp2:
    #     single = []
    #     single.append(p.pt[0])
    #     single.append(p.pt[1])
    #     individual.append(single)
    # outer.append(individual)
    # numpy_array = np.asarray(outer)
    # print(numpy_array)


    # for p in kp2:
    #     print(p.pt)

    if len(kp1) == 0:
        kp1, last_frame = kp2, resized_frame
        ret, frame = cap.read()
        continue;
    else:
        kp2, st, err = cv.calcOpticalFlowPyrLK(last_frame, resized_frame, kp1, None, **lk_params)

    good_new = kp2[st==1]
    good_old = kp1[st==1]

    print(len(good_new))    

    F, mask = cv.findFundamentalMat(good_new, good_old, cv.FM_RANSAC, 10, 0.99)
    # F, mask = cv.findFundamentalMat(good_old,good_new,cv.FM_8POINT)

    # print(F)

    # E, _ = cv.findEssentialMat(good_old,good_new,718.856,(607.1928,185.2157),cv.RANSAC,0.99,1.0)
    E, _ = cv.findEssentialMat(good_old, good_new,F, cv.RANSAC,0.99,1.0,mask)
    # E, _ = cv.findEssentialMat(good_old,good_new,F,method=None,prob=0.99,threshold=1.0,mask=mask)
    # print(E)

    _, R, t, _ = cv.recoverPose(E, good_old, good_new, F, R.copy(), t.copy(), None)

    print(t)

    cv.imshow('frame', resized_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv.destroyAllWindows()