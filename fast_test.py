import numpy as np
import cv2 as cv

cap = cv.VideoCapture('./video/testVideo.mp4')
ret,frame = cap.read()

height, width = frame.shape[:2]

W = int(width / 4)
H = int(height / 4)

fast = cv.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

R = np.zeros(shape=(3,3))
t = np.zeros(shape=(3,3))

lk_params = dict(winSize = (15,15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 3000,
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
    return fast.detect(frame, None)

def asnumpyarray(keypoints):
    l = []
    for p in keypoints:
        l.append([p.pt[0], p.pt[1]])
    return np.asarray(l)

while True:
    resized_frame = cv.resize(frame, (W, H))
    resized_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2GRAY)

    mask = np.zeros_like(resized_frame)

    # kp2 = process_features(resized_frame)
    kp2 = process_good_features(resized_frame)

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
        continue;
    else:
        kp2, st, err = cv.calcOpticalFlowPyrLK(last_frame, resized_frame, kp1, None, **lk_params)

    good_new = kp2[st==1]
    good_old = kp1[st==1]

    new_frame = []

    for i, (new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b), (c,d), color[i % len(color)].tolist(), 2)
        new_frame = cv.circle(resized_frame,(a,b),5,color[i % len(color)].tolist(),-1)
    
    img = cv.add(new_frame, mask)
    # for i in kp2:
    #     x,y = i.ravel()
    #     img = cv.circle(resized_frame,(x,y),3,255,-1)

    # img = cv.drawKeypoints(resized_frame,kp,outImage=None,color=(255,0,0),flags=None)

    # new_frame = cv.cvtColor(new_frame, cv.COLOR_GRAY2RGB)

    cv.imshow('frame', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv.destroyAllWindows()