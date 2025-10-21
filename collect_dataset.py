# collect_dataset.py
import argparse
import csv
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ========================= Helpers (geometry & mask) =========================
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


def filter_close_points(points, min_dist=30):
    filtered = []
    for p in points:
        p_arr = np.array(p)
        if all(np.linalg.norm(p_arr - np.array(q)) > min_dist for q in filtered):
            filtered.append(tuple(p_arr))
    return filtered


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

    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


# ========================= Tip extraction (from crop) =========================
def extract_tips_from_crop(crop_bgr,
                           angle_th=80, depth_ratio=0.06,
                           y_above=20, tip_min_dist=30):
    """
    คืนค่า:
      tips_local: list[(x,y)] ในพิกัด local ของ crop (ซ้าย->ขวา)
      wrist_local: tuple(x,y) กึ่งกลางล่างของ bbox
    """
    h, w = crop_bgr.shape[:2]
    mask = combined_skin_mask(crop_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], (w // 2, h)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return [], (w // 2, h)

    wrist_local = (w // 2, h)

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

    if not tips:
        tips = [tuple(p) for p in hull_pts if p[1] < wrist_local[1] - y_above]

    tips = filter_close_points(tips, min_dist=tip_min_dist)
    tips = sorted(tips, key=lambda p: p[0])  # left->right
    return tips, wrist_local


# ========================= Feature builder =========================
def build_feature_vector(tips_local, wrist_local, bbox_w, bbox_h, max_tips=5):
    """
    สร้างเวกเตอร์ฟีเจอร์คงที่ความยาว:
      [finger_count,
       tip1_x_norm, tip1_y_norm, tip1_r_norm, tip1_theta,
       ... (ซ้ำถึง tip5),
      ]
    - x_norm,y_norm: normalized ด้วย bbox_w,bbox_h (0..1)
    - r_norm: ระยะจาก wrist / max(bbox_w,bbox_h)
    - theta: มุมเชิงขั้ว (radians) มุมจาก wrist -> tip, ช่วง [-pi, pi]
    ถ้าจำนวน tips < max_tips จะ pad ด้วย 0
    """
    features = []
    finger_count = len(tips_local)
    features.append(finger_count)

    M = max(bbox_w, bbox_h, 1)  # ป้องกันหาร 0
    for i in range(max_tips):
        if i < finger_count:
            tx, ty = tips_local[i]
            x_norm = tx / max(bbox_w, 1)
            y_norm = ty / max(bbox_h, 1)
            dx = tx - wrist_local[0]
            dy = ty - wrist_local[1]
            r_norm = (math.hypot(dx, dy)) / M
            theta = math.atan2(dy, dx)  # [-pi, pi]
        else:
            x_norm = y_norm = r_norm = theta = 0.0

        features.extend([x_norm, y_norm, r_norm, theta])

    return features  # length = 1 + max_tips*4


# ========================= Main (collection loop) =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Collect hand gesture dataset (YOLO + CV, no MediaPipe)")
    ap.add_argument("--model", type=str, default="yolov8n.pt",
                    help="Path to YOLO .pt model (hand model recommended).")
    ap.add_argument("--device", type=str, default="cpu",
                    help='Device: "cpu" or "0" for CUDA:0, etc.')
    ap.add_argument("--imgsz", type=int, default=640,
                    help="YOLO inference size.")
    ap.add_argument("--cam", type=int, default=0,
                    help="Camera index.")
    ap.add_argument("--min_bbox_area", type=int, default=8000,
                    help="Minimum bbox area to accept.")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="Confidence threshold for YOLO.")
    ap.add_argument("--csv", type=str, default="dataset.csv",
                    help="Output CSV file path.")
    ap.add_argument("--imgdir", type=str, default="dataset_imgs",
                    help="Directory to save crops (optional).")
    ap.add_argument("--classes", type=str, default="open,closed,peace,rock,point",
                    help="Comma-separated class names mapped to keys 1..N (e.g., 'open,closed').")
    ap.add_argument("--save_images", action="store_true",
                    help="If set, save crop images per sample.")
    return ap.parse_args()


def main():
    args = parse_args()

    # Prepare classes & key map
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    key_to_label = {ord(str(i+1)): class_names[i] for i in range(min(9, len(class_names)))}
    current_label = class_names[0] if class_names else "unknown"

    # Init paths
    csv_path = Path(args.csv)
    img_root = Path(args.imgdir)
    if args.save_images:
        img_root.mkdir(parents=True, exist_ok=True)

    # Init CSV header if not exists
    header = ["timestamp", "label", "finger_count"]
    # 5 tips * (x_norm,y_norm,r_norm,theta)
    for i in range(1, 6):
        header += [f"tip{i}_x_norm", f"tip{i}_y_norm", f"tip{i}_r_norm", f"tip{i}_theta"]

    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Load YOLO model
    model = YOLO(args.model)
    hand_class_ids = [k for k, v in model.names.items()
                      if 'hand' in v.lower() or 'left_hand' in v.lower() or 'right_hand' in v.lower()]
    print("Classes:", model.names)
    print("Hand-class ids:", hand_class_ids)
    print(f"Labels (press number keys): { {i+1: name for i,name in enumerate(class_names)} }")
    print(f"Current label: {current_label}")
    print("Press number [1..9] to switch label, 'S' to save sample, 'Q' to quit.")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    sample_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # YOLO inference
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device)
        r = results[0] if results else None
        boxes = r.boxes if r is not None else None

        display = frame.copy()
        chosen_bbox = None
        tips_local, wrist_local = [], (0, 0)

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = (boxes.cls.cpu().numpy().astype(int)
                    if boxes.cls is not None else [None] * len(confs))

            # เลือก bbox ที่ตรง class (ถ้ามี hand) หรือ 'person' (fallback) ที่ใหญ่สุด
            best_area = 0
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                w, h = (x2 - x1), (y2 - y1)
                area = w * h
                if area < args.min_bbox_area or w <= 0 or h <= 0:
                    continue

                label = model.names.get(int(cls), str(cls)) if cls is not None else "obj"
                good = False
                if hand_class_ids:
                    good = int(cls) in hand_class_ids
                else:
                    good = (label.lower() == "person")

                if good and area > best_area:
                    best_area = area
                    chosen_bbox = (x1, y1, x2, y2)

            # มี bbox ที่เลือก → extract tips
            if chosen_bbox is not None:
                x1, y1, x2, y2 = chosen_bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crop = frame[y1:y2, x1:x2].copy()
                tips_local, wrist_local = extract_tips_from_crop(crop)
                # วาด tips preview
                for (tx, ty) in tips_local:
                    cv2.circle(display, (x1 + int(tx), y1 + int(ty)), 4, (0, 0, 255), -1)
                # วาด wrist
                cv2.circle(display, (x1 + wrist_local[0], y1 + wrist_local[1]), 4, (255, 255, 255), -1)

        # HUD
        cv2.putText(display, f"Label: {current_label}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2)
        cv2.putText(display, "Keys: [1..9]=label  S=save  Q=quit", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display, f"Tips: {len(tips_local)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

        cv2.imshow("Collect Dataset (YOLO+CV, no MediaPipe)", display)
        key = cv2.waitKey(1) & 0xFF

        # เปลี่ยน label ด้วยปุ่มเลข
        if key in key_to_label:
            current_label = key_to_label[key]
            print(f">> Current label = {current_label}")

        # บันทึก sample
        if key in (ord('s'), ord('S')):
            if chosen_bbox is None:
                print("!! No bbox selected, cannot save.")
                continue

            x1, y1, x2, y2 = chosen_bbox
            bbox_w, bbox_h = (x2 - x1), (y2 - y1)

            # build feature vector (fixed length)
            feats = build_feature_vector(tips_local, wrist_local, bbox_w, bbox_h, max_tips=5)
            finger_count = feats[0]

            # แถวข้อมูล
            row = [int(time.time() * 1000), current_label, finger_count] + feats[1:]
            with csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # บันทึกรูป (ถ้าเปิด)
            if args.save_images:
                label_dir = img_root / current_label
                label_dir.mkdir(parents=True, exist_ok=True)
                sample_idx += 1
                img_path = label_dir / f"{int(time.time()*1000)}_{sample_idx}.jpg"
                crop_bgr = frame[y1:y2, x1:x2]
                cv2.imwrite(str(img_path), crop_bgr)

            print(f"✓ Saved sample: label={current_label}, tips={len(tips_local)}")

        # ออก
        if key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
