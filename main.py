import numpy as np
import cv2 as cv

from matcher import Frame, get_pose, get_features, denormalize

cap = cv.VideoCapture('./video/testVideo.mp4')
ret, frame = cap.read()

height, width = frame.shape[:2]

# camera intrinsics
W = int(width / 4)
H = int(height / 4)
F = 272.93867730682706
K = np.array([[F,0,W//2], [0,F,H//2], [0,0,1]])

frames = []

IRt = np.eye(3)
while True:
    original_frame = cv.resize(frame, (W, H))
    resized_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY)
    
    frame = Frame(resized_frame, K)
    frames.append(frame)

    if len(frames) <= 1:
        ret, frame = cap.read()
        continue;

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

        img = cv.circle(original_frame, (u1,v1), color=(255,255,0), radius=3)
        cv.line(original_frame, (u1,v1), (u2,v2), color=(255,0,0))

    cv.imshow('frame', img)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    ret, frame = cap.read()

cap.release()
cv.destroyAllWindows()