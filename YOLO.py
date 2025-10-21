# yolov_integration_hand.py
import argparse
import math
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------- Helpers: geometry & filtering -----------------
def angle_between(a, mid, b):
    a, mid, b = map(np.array, (a, mid, b))
    v1 = a - mid
    v2 = b - mid
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 180.0
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def point_line_distance(start, end, point):
    start, end, point = map(np.array, (start, end, point))
    line = end - start
    if np.all(line == 0):
        return np.linalg.norm(point - start)
    area = abs(np.cross(line, point - start))
    return area / np.linalg.norm(line)


def filter_close_points(points, min_dist=40):
    filtered = []
    for p in points:
        p_arr = np.array(p)
        if all(np.linalg.norm(p_arr - np.array(q)) > min_dist for q in filtered):
            filtered.append(tuple(p_arr))
    return filtered


def get_landmarks(wrist, tip):
    wrist, tip = np.array(wrist, dtype=float), np.array(tip, dtype=float)
    mcp = tuple(np.int32(wrist + 0.25 * (tip - wrist)))
    pip = tuple(np.int32(wrist + 0.55 * (tip - wrist)))
    dip = tuple(np.int32(wrist + 0.8 * (tip - wrist)))
    return mcp, pip, dip, tuple(np.int32(tip))


# ----------------- Skin mask (HSV + YCrCb) -----------------
def combined_skin_mask(img, hsv_range=None, ycrcb_range=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    if hsv_range is None:
        lower_hsv = np.array([0, 12, 50], dtype=np.uint8)
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    else:
        lower_hsv, upper_hsv = hsv_range

    if ycrcb_range is None:
        lower_y = np.array([0, 133, 77], dtype=np.uint8)
        upper_y = np.array([255, 173, 127], dtype=np.uint8)
    else:
        lower_y, upper_y = ycrcb_range

    mask1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask2 = cv2.inRange(ycrcb, lower_y, upper_y)
    mask = cv2.bitwise_and(mask1, mask2)

    # denoise
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


# ----------------- Process one crop (inside YOLO bbox) -----------------
def process_hand_crop_and_draw(orig_frame, crop, x_off, y_off,
                               angle_th=80, depth_ratio=0.06, y_above=20,
                               tip_min_dist=30):
    """
    crop: BGR image of bbox area
    (x_off, y_off): offsets to map crop coords back to original frame
    """
    h, w = crop.shape[:2]
    mask = combined_skin_mask(crop)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return orig_frame, 0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 500:  # too small within crop
        return orig_frame, 0

    # approximate wrist at bottom-center of crop
    wrist_local = (w // 2, h)

    # hull & defects
    hull_pts = cv2.convexHull(largest, returnPoints=True)[:, 0, :]
    hull_idx = cv2.convexHull(largest, returnPoints=False)[:, 0] if len(largest) >= 3 else None

    tips = []
    if hull_idx is not None and len(hull_idx) > 3:
        defects = cv2.convexityDefects(largest, hull_idx)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, _ = defects[i, 0]
                spt = tuple(largest[s][0])
                ept = tuple(largest[e][0])
                fpt = tuple(largest[f][0])
                geom_depth = point_line_distance(spt, ept, fpt)
                ang = angle_between(spt, fpt, ept)
                if ang < angle_th and geom_depth > (h * depth_ratio):
                    if spt[1] < wrist_local[1] - y_above:
                        tips.append(spt)
                    if ept[1] < wrist_local[1] - y_above:
                        tips.append(ept)

    # fallback: hull points above wrist
    if not tips:
        tips = [tuple(p) for p in hull_pts if p[1] < wrist_local[1] - y_above]

    tips = filter_close_points(tips, min_dist=tip_min_dist)
    tips = sorted(tips, key=lambda p: p[0])

    # draw skeleton (map local->global)
    for tip in tips:
        tip_global = (int(tip[0] + x_off), int(tip[1] + y_off))
        wrist_global = (int(wrist_local[0] + x_off), int(wrist_local[1] + y_off))
        mcp, pip, dip, tip_point = get_landmarks(wrist_global, tip_global)

        for point, c in zip([mcp, pip, dip, tip_point],
                            [(0, 255, 255), (0, 165, 255), (0, 100, 255), (0, 0, 255)]):
            cv2.circle(orig_frame, point, 4, c, -1)

        cv2.line(orig_frame, wrist_global, mcp, (255, 255, 255), 2)
        cv2.line(orig_frame, mcp, pip, (255, 255, 255), 2)
        cv2.line(orig_frame, pip, dip, (255, 255, 255), 2)
        cv2.line(orig_frame, dip, tip_point, (255, 255, 255), 2)

    return orig_frame, len(tips)


# ----------------- Main runtime -----------------
def run_with_yolo(
    yolo_model_path: str,
    conf_thres: float = 0.35,
    imgsz: int = 640,
    device: str = "cpu",
    cam_index: int = 0,
    min_bbox_area: int = 8000,
):
    """
    yolo_model_path: path to .pt model (e.g., hand_yolo.pt or yolov8n.pt)
    conf_thres: detection confidence threshold
    imgsz: YOLO inference size
    device: "cpu" or "0" (CUDA:0) etc.
    cam_index: camera index for OpenCV
    min_bbox_area: filter out tiny bboxes
    """
    model = YOLO(yolo_model_path)
    print("Classes:", model.names)

    # Try to detect if model has hand classes
    hand_class_ids = [k for k, v in model.names.items()
                      if 'hand' in v.lower() or 'left_hand' in v.lower() or 'right_hand' in v.lower()]
    print("Hand-class ids:", hand_class_ids)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Running. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # YOLO inference (single image)
        results = model.predict(frame, imgsz=imgsz, conf=conf_thres, device=device)

        if not results:
            cv2.imshow("YOLO + CV Hand", frame)
        else:
            r = results[0]
            boxes = r.boxes
            if boxes is None:
                cv2.imshow("YOLO + CV Hand", frame)
            else:
                xyxy = boxes.xyxy.cpu().numpy()    # Nx4
                confs = boxes.conf.cpu().numpy()   # N
                clss = (boxes.cls.cpu().numpy().astype(int)
                        if boxes.cls is not None else [None] * len(confs))

                for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    w, h = (x2 - x1), (y2 - y1)
                    if w <= 0 or h <= 0:
                        continue
                    area = w * h
                    if area < min_bbox_area:
                        continue

                    label = model.names.get(int(cls), str(cls)) if cls is not None else "obj"

                    # Filtering logic:
                    # 1) If model has hand classes -> keep only hand
                    # 2) Else (e.g., yolov8n.pt on COCO) -> keep only 'person'
                    if hand_class_ids:
                        if int(cls) not in hand_class_ids:
                            continue
                    else:
                        if label.lower() != "person":
                            continue

                    # Draw bbox, process crop
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    crop = frame[y1:y2, x1:x2].copy()
                    frame, tip_count = process_hand_crop_and_draw(frame, crop, x1, y1)

                    # quick heuristic label for open/closed (optional)
                    status = "OPEN" if tip_count >= 3 else "CLOSED"
                    cv2.putText(frame, f"{label} {conf:.2f} tips:{tip_count} [{status}]",
                                (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)

                cv2.imshow("YOLO + CV Hand", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="YOLO (hand/person) + CV hand skeleton (no MediaPipe)")
    ap.add_argument("--model", type=str, default="yolov8n.pt",
                    help="Path to YOLO .pt model (hand model recommended)")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="Confidence threshold (default: 0.35)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO image size (default: 640)")
    ap.add_argument("--device", type=str, default="cpu",
                    help='Device: "cpu" or "0" (CUDA:0) etc. (default: cpu)')
    ap.add_argument("--cam", type=int, default=0,
                    help="Camera index (default: 0)")
    ap.add_argument("--min_bbox_area", type=int, default=8000,
                    help="Filter tiny bbox area (default: 8000)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_with_yolo(
        yolo_model_path=args.model,
        conf_thres=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        cam_index=args.cam,
        min_bbox_area=args.min_bbox_area,
    )
