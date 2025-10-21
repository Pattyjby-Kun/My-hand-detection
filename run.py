# run_gesture.py
import argparse, math, joblib
import cv2, numpy as np
from ultralytics import YOLO
from collections import deque, Counter

# -------- helpers --------
def angle_between(a, mid, b):
    a, mid, b = map(np.array, (a, mid, b))
    v1, v2 = a - mid, b - mid
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    if d == 0: return 180.0
    c = np.clip(np.dot(v1, v2) / d, -1.0, 1.0)
    return math.degrees(math.acos(c))

def point_line_distance(start, end, point):
    start, end, point = map(np.array, (start, end, point))
    line = end - start
    if np.all(line == 0): return np.linalg.norm(point - start)
    area = abs(np.cross(line, point - start))
    return area / np.linalg.norm(line)

def filter_close_points(points, min_dist=30):
    out = []
    for p in points:
        p = np.array(p)
        if all(np.linalg.norm(p - np.array(q)) > min_dist for q in out):
            out.append(tuple(p))
    return out

def combined_skin_mask(img, hsv_range=None, ycrcb_range=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_hsv = np.array([0, 12, 50], np.uint8) if not hsv_range else hsv_range[0]
    upper_hsv = np.array([25, 255, 255], np.uint8) if not hsv_range else hsv_range[1]
    lower_y   = np.array([0, 133, 77], np.uint8) if not ycrcb_range else ycrcb_range[0]
    upper_y   = np.array([255, 173, 127], np.uint8) if not ycrcb_range else ycrcb_range[1]
    m1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
    m2 = cv2.inRange(ycrcb, lower_y, upper_y)
    m  = cv2.bitwise_and(m1, m2)
    m  = cv2.GaussianBlur(m, (7, 7), 0)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m  = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m  = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m

def extract_tips_from_crop(crop_bgr, angle_th=80, depth_ratio=0.06, y_above=20, tip_min_dist=30):
    h, w = crop_bgr.shape[:2]
    mask = combined_skin_mask(crop_bgr)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return [], (w//2, h)
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500: return [], (w//2, h)
    wrist_local = (w//2, h)
    hull_pts = cv2.convexHull(largest, returnPoints=True)[:,0,:]
    hull_idx = cv2.convexHull(largest, returnPoints=False)[:,0] if len(largest)>=3 else None
    tips=[]
    if hull_idx is not None and len(hull_idx)>3:
        defects = cv2.convexityDefects(largest, hull_idx)
        if defects is not None:
            for s,e,f,_ in defects[:,0,:]:
                spt = tuple(largest[s][0]); ept = tuple(largest[e][0]); fpt = tuple(largest[f][0])
                depth = point_line_distance(spt,ept,fpt)
                ang   = angle_between(spt,fpt,ept)
                if ang < angle_th and depth > (h*depth_ratio):
                    if spt[1] < wrist_local[1]-y_above: tips.append(spt)
                    if ept[1] < wrist_local[1]-y_above: tips.append(ept)
    if not tips:
        tips = [tuple(p) for p in hull_pts if p[1] < wrist_local[1]-y_above]
    tips = filter_close_points(tips, tip_min_dist)
    return sorted(tips, key=lambda p:p[0]), wrist_local

def build_feature_vector(tips_local, wrist_local, bbox_w, bbox_h, max_tips=5):
    feats=[len(tips_local)]
    M=max(bbox_w,bbox_h,1)
    for i in range(max_tips):
        if i < len(tips_local):
            tx,ty = tips_local[i]
            x_norm = tx/max(bbox_w,1); y_norm=ty/max(bbox_h,1)
            dx,dy = tx-wrist_local[0], ty-wrist_local[1]
            r_norm = (dx*dx+dy*dy)**0.5 / M
            theta  = math.atan2(dy,dx)
        else:
            x_norm=y_norm=r_norm=theta=0.0
        feats += [x_norm,y_norm,r_norm,theta]
    return np.array(feats, dtype=float)

def get_landmarks(wrist, tip):
    wrist, tip = np.array(wrist, dtype=float), np.array(tip, dtype=float)
    mcp = tuple(np.int32(wrist + 0.25 * (tip - wrist)))
    pip = tuple(np.int32(wrist + 0.55 * (tip - wrist)))
    dip = tuple(np.int32(wrist + 0.8 * (tip - wrist)))
    return mcp, pip, dip, tuple(np.int32(tip))

# -------- runtime --------
def parse_args():
    ap = argparse.ArgumentParser(description="Run real-time gesture classification")
    ap.add_argument("--model", type=str, default="d033f814-704c-4997-bee6-e03f3acaf037.pt", help="YOLO .pt (hand model preferred)")
    ap.add_argument("--clf", type=str, default="gesture_model.pkl", help="Trained classifier (joblib)")
    ap.add_argument("--device", type=str, default="cpu", help='YOLO device: "cpu" or "0"')
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--min_bbox_area", type=int, default=6000)
    ap.add_argument("--conf", type=float, default=0.35)
    return ap.parse_args()

def main():
    args = parse_args()
    clf_pack = joblib.load(args.clf)
    pipe = clf_pack["pipeline"]
    le   = clf_pack["label_encoder"]

    yolo = YOLO(args.model)
    names = yolo.names
    hand_ids = [k for k,v in names.items() if "hand" in v.lower() or "left_hand" in v.lower() or "right_hand" in v.lower()]
    print("YOLO classes:", names)
    print("Hand ids:", hand_ids)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("❌ Cannot open camera"); return
    print("✅ Running. Press 'q' to quit.")

    pred_hist = deque(maxlen=5)

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)

        res = yolo.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device)
        r = res[0] if res else None
        boxes = r.boxes if r is not None else None

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else [None]*len(confs)

            best=None; best_area=0
            for (x1,y1,x2,y2), c, cls in zip(xyxy, confs, clss):
                x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
                w,h = x2-x1, y2-y1
                if w<=0 or h<=0: continue
                area = w*h
                if area < args.min_bbox_area: continue
                label = names.get(int(cls), str(cls)) if cls is not None else "obj"
                good = (int(cls) in hand_ids) if hand_ids else (label.lower()=="person")
                if good and area>best_area:
                    best = (x1,y1,x2,y2); best_area=area

            if best:
                x1,y1,x2,y2 = best
                crop = frame[y1:y2, x1:x2]
                tips, wrist = extract_tips_from_crop(crop)
                feats = build_feature_vector(tips, wrist, x2-x1, y2-y1, max_tips=5).reshape(1,-1)
                pred_idx = pipe.predict(feats)[0]
                pred_lbl = le.inverse_transform([pred_idx])[0]
                pred_hist.append(pred_lbl)
                if len(pred_hist)>0:
                    pred_lbl = Counter(pred_hist).most_common(1)[0][0]

                # ---- draw skeleton ----
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                wrist_global = (x1 + wrist[0], y1 + wrist[1])
                for tip in tips:
                    tip_global = (x1+int(tip[0]), y1+int(tip[1]))
                    mcp, pip, dip, tip_point = get_landmarks(wrist_global, tip_global)
                    for point, c in zip([mcp, pip, dip, tip_point],
                                        [(0,255,255),(0,165,255),(0,100,255),(0,0,255)]):
                        cv2.circle(frame, point, 4, c, -1)
                    cv2.line(frame, wrist_global, mcp, (255,255,255), 2)
                    cv2.line(frame, mcp, pip, (255,255,255), 2)
                    cv2.line(frame, pip, dip, (255,255,255), 2)
                    cv2.line(frame, dip, tip_point, (255,255,255), 2)
                cv2.circle(frame, wrist_global, 6, (255,255,255), -1)

                cv2.putText(frame, f"Gesture: {pred_lbl}", (x1, max(10,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Gesture Runtime", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
