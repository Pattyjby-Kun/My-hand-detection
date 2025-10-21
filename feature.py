# feature.py
import numpy as np
from collections import deque

# --- indices (ตาม MyMediapipe ordering: tip at 0,4,8,12,16 and wrist as last index) ---
def fingertip_indices():
    return [0, 4, 8, 12, 16]

def fingertip_points(landmarks):
    inds = fingertip_indices()
    return np.array([landmarks[i] for i in inds])  # shape (5,2)

def wrist_point(landmarks):
    return np.array(landmarks[-1])  # wrist assumed last

# normalization base: choose hand size (max of bbox w,h) or palm-to-tip avg
def hand_size_from_bbox(bbox):
    x,y,w,h = bbox
    return float(max(w, h))

def palm_to_tip_mean(landmarks, bbox):
    palm = wrist_point(landmarks)
    tips = fingertip_points(landmarks)
    d = np.linalg.norm(tips - palm, axis=1)
    return float(np.mean(d))

# --- Basic features ---
def finger_pair_distance_norm(landmarks, bbox, norm_by='bbox'):
    """
    Return normalized distances between adjacent fingertips (4 values)
    norm_by: 'bbox' -> divide by max(w,h)
             'palm' -> divide by mean palm-to-tip distance
    """
    tips = fingertip_points(landmarks)
    dists = []
    for i in range(len(tips)-1):
        d = np.linalg.norm(tips[i] - tips[i+1])
        dists.append(d)
    dists = np.array(dists, dtype=float)
    if norm_by == 'palm':
        denom = palm_to_tip_mean(landmarks, bbox) + 1e-9
    else:
        denom = hand_size_from_bbox(bbox) + 1e-9
    return dists / denom  # length 4

def tip_to_palm_norm(landmarks, bbox, norm_by='bbox'):
    tips = fingertip_points(landmarks)
    palm = wrist_point(landmarks)
    d = np.linalg.norm(tips - palm, axis=1)
    if norm_by == 'palm':
        denom = np.mean(d) + 1e-9
    else:
        denom = hand_size_from_bbox(bbox) + 1e-9
    return (d / denom)  # length 5

def finger_angles(landmarks):
    """
    Angle (radians) of vector palm->tip in image coords, [-pi,pi]
    Useful to check finger directions/orientation
    """
    tips = fingertip_points(landmarks)
    palm = wrist_point(landmarks)
    vecs = tips - palm
    angles = np.arctan2(vecs[:,1], vecs[:,0])
    return angles  # length 5

# --- Decision: Open / Closed / Touching ---
def default_thresholds():
    return {
        'pair_thresh': 0.08,    # normalized pairwise distance threshold (start value)
        'pair_hyst': (0.07, 0.10),  # enter, exit thresholds for hysteresis (lower->enter touch, higher->exit touch)
        'tip_thresh': 0.45,     # normalized tip-to-palm threshold for open/closed heuristic (start)
        'tip_hyst': (0.40, 0.50)
    }

def finger_states(landmarks, bbox, thresholds=None, norm_by='bbox'):
    """
    Return list of dicts per finger pair and per finger:
     - pair: {pair:(i,i+1), norm_dist, touching_bool}
     - finger: {finger_idx, tip_palm_norm, is_open_bool}
    """
    if thresholds is None:
        thresholds = default_thresholds()
    # pair distances (len 4)
    pair_d = finger_pair_distance_norm(landmarks, bbox, norm_by=norm_by)
    tip_norm = tip_to_palm_norm(landmarks, bbox, norm_by=norm_by)
    angles = finger_angles(landmarks)

    # touching mask using simple threshold (no hysteresis here)
    pair_touch = pair_d < thresholds['pair_thresh']

    # finger open/closed: if tip_to_palm normalized > tip_thresh -> open else closed
    finger_open = tip_norm > thresholds['tip_thresh']

    pair_results = []
    for i, val in enumerate(pair_d):
        pair_results.append({
            'pair': (i, i+1),
            'norm_dist': float(val),
            'touching': bool(pair_touch[i])
        })

    finger_results = []
    for i, val in enumerate(tip_norm):
        finger_results.append({
            'finger': i,
            'tip_palm_norm': float(val),
            'angle': float(angles[i]),
            'is_open': bool(finger_open[i])
        })

    return {'pairs': pair_results, 'fingers': finger_results}

# --- Hysteresis + temporal smoothing helper class ---
class HysteresisFilter:
    def __init__(self, enter_thresh, exit_thresh, initial=False):
        # enter_thresh: value threshold to switch to True (e.g., val < enter -> touching)
        # exit_thresh: threshold to switch to False
        self.enter = enter_thresh
        self.exit = exit_thresh
        self.state = initial

    def update_touch(self, val):
        # for pair distances where smaller means touch:
        if not self.state and val < self.enter:
            self.state = True
        elif self.state and val > self.exit:
            self.state = False
        return self.state

class TemporalSmoother:
    def __init__(self, n_frames=5):
        self.n = n_frames
        # per pair and per finger deques will be created at first update
        self.pair_hist = None
        self.finger_hist = None

    def init_lengths(self, n_pairs=4, n_fingers=5):
        self.pair_hist = [deque(maxlen=self.n) for _ in range(n_pairs)]
        self.finger_hist = [deque(maxlen=self.n) for _ in range(n_fingers)]

    def update(self, pair_bool_list, finger_bool_list):
        if self.pair_hist is None:
            self.init_lengths(n_pairs=len(pair_bool_list), n_fingers=len(finger_bool_list))
        out_pairs = []
        out_fingers = []
        for i,b in enumerate(pair_bool_list):
            self.pair_hist[i].append(bool(b))
            # majority vote
            out_pairs.append(sum(self.pair_hist[i]) > (len(self.pair_hist[i]) / 2.0))
        for i,b in enumerate(finger_bool_list):
            self.finger_hist[i].append(bool(b))
            out_fingers.append(sum(self.finger_hist[i]) > (len(self.finger_hist[i]) / 2.0))
        return out_pairs, out_fingers
