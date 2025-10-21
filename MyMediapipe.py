# MyMediapipe.py
import numpy as np
import cv2
from math import atan2, hypot, pi

def contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return np.array([0,0], dtype=np.float32)
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']
    return np.array([cx, cy], dtype=np.float32)

def angle_of(p, center):
    return atan2(p[1]-center[1], p[0]-center[0])

class MyMediapipe:
    def __init__(self, n_fingers=5):
        self.n_fingers = n_fingers

    def get_landmarks(self, contour, bbox):
        """
        Input: contour (numpy array), bbox (x,y,w,h)
        Output: landmarks: list of 21 (x,y) in image coords (float)
                ordering: for each finger: [tip, dip, pip, mcp] (4*5=20) then wrist as 21st
        """
        # compute palm center
        palm = contour_centroid(contour)
        # compute convex hull points
        hull = cv2.convexHull(contour, returnPoints=True)[:,0,:]  # (N,2)
        # compute distances and angles relative to palm
        pts = hull
        angles = np.array([angle_of(p, palm) for p in pts])
        dists = np.array([hypot(p[0]-palm[0], p[1]-palm[1]) for p in pts])
        # bin angles into n_fingers sectors
        bins = np.linspace(-pi, pi, self.n_fingers+1)
        fingertip_candidates = []
        for i in range(self.n_fingers):
            mask = (angles >= bins[i]) & (angles < bins[i+1])
            if not np.any(mask):
                fingertip_candidates.append(None)
            else:
                idx = np.argmax(dists * mask)  # picks farthest in that sector
                # but argmax above gives index in whole array; instead:
                sector_idx = np.where(mask)[0]
                far_idx = sector_idx[np.argmax(dists[sector_idx])]
                fingertip_candidates.append(tuple(pts[far_idx]))

        # clean None and ensure 5 points: try fallback pick top-5 farthest points sorted by angle
        tips = []
        for cand in fingertip_candidates:
            if cand is None:
                # fallback: choose farthest remaining
                pass
            tips.append(np.array(cand) if cand is not None else None)
        # If any None, pick additional farthest points to fill
        if any(t is None for t in tips):
            order = np.argsort(-dists)
            filled = []
            for idx in order:
                p = pts[idx]
                # accept if angle not close to existing tips
                a = angle_of(p, palm)
                if all((t is None) or (abs(angle_of(t,palm)-a)>0.3) for t in tips):
                    filled.append(tuple(p))
                if len(filled) >= sum(1 for t in tips if t is None):
                    break
            fi = 0
            for i,t in enumerate(tips):
                if t is None:
                    tips[i] = np.array(filled[fi]); fi+=1

        # now tips has 5 points
        # order fingertips by x or angle to be consistent: sort by angle
        tips = np.array(tips)
        tip_angles = np.array([angle_of(t,palm) for t in tips])
        order_idx = np.argsort(tip_angles)  # left->right depending on orientation
        tips = tips[order_idx]

        # build joints along vector tip -> palm at fractions (tip, dip, pip, mcp)
        fractions = [0.0, 0.2, 0.45, 0.75]  # proportion from tip towards palm
        landmarks = []
        for t in tips:
            v = palm - t
            finger_pts = [t + frac*v for frac in fractions]
            landmarks.extend(finger_pts)

        # wrist approximate: use bbox bottom center
        x,y,w,h = bbox
        wrist = np.array([x + w/2, y + h*0.95])
        landmarks.append(wrist)
        # return as float Nx2 array
        return np.array(landmarks, dtype=np.float32)  # shape (21,2)

def draw_landmarks(img, landmarks, color=(0,255,0)):
    for i,(x,y) in enumerate(landmarks.astype(int)):
        cv2.circle(img, (x,y), 4, color, -1)
    # draw simple finger lines (per finger 4 points)
    for f in range(5):
        pts = landmarks[f*4:(f+1)*4].astype(int)
        for i in range(len(pts)-1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), color, 2)
    # draw wrist
    cv2.circle(img, tuple(landmarks[-1].astype(int)), 5, (0,0,255), -1)
