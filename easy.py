# improved_hand_skeleton_no_mediapipe.py
import cv2
import numpy as np
from collections import deque
import math

# ---- พารามิเตอร์ที่ปรับได้ ----
MIN_AREA = 1500            # พื้นที่ contour ขั้นต่ำ (ปรับตามขนาดกล้อง/ระยะมือ)
MIN_DIST = 40              # ระยะห่างขั้นต่ำสำหรับกรองปลายๆ ที่ซ้อนกัน
SMOOTH_LEN = 5             # จำนวนเฟรมสำหรับ moving-average smoothing
DEFECT_ANGLE_TH = 80       # มุม (deg) ที่ถือว่าเป็น gap ระหว่างนิ้ว (ยิ่งเล็ก = เข้มงวด)
DEFECT_DEPTH_RATIO = 0.08  # ค่าระยะเชิงสัมพัทธ์ (depth > h * ratio) เพื่อกรอง defect เล็กๆ

# ฟังก์ชันคำนวณจุด landmark (MCP, PIP, DIP)
def get_landmarks(wrist, tip):
    wrist, tip = np.array(wrist, dtype=float), np.array(tip, dtype=float)
    mcp = tuple(np.int32(wrist + 0.25 * (tip - wrist)))
    pip = tuple(np.int32(wrist + 0.55 * (tip - wrist)))
    dip = tuple(np.int32(wrist + 0.8 * (tip - wrist)))
    return mcp, pip, dip, tuple(np.int32(tip))

# กรองจุดที่ใกล้กันเกินไป
def filter_close_points(points, min_dist=MIN_DIST):
    filtered = []
    for p in points:
        p_arr = np.array(p)
        if all(np.linalg.norm(p_arr - np.array(q)) > min_dist for q in filtered):
            filtered.append(tuple(p_arr))
    return filtered

# มุมระหว่างสามจุด (deg) — angle at 'mid'
def angle_between(a, mid, b):
    a, mid, b = map(np.array, (a, mid, b))
    v1 = a - mid
    v2 = b - mid
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 180.0
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

# ระยะจากจุด far ไปยังเส้น (start-end)
def point_line_distance(start, end, point):
    start, end, point = map(np.array, (start, end, point))
    line = end - start
    if np.all(line == 0):
        return np.linalg.norm(point - start)
    area = abs(np.cross(line, point - start))
    return area / np.linalg.norm(line)

def improved_hand_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้")
        return

    tips_history = deque(maxlen=SMOOTH_LEN)

    print("✅ กล้องทำงานปกติ — แสดงมือต่อหน้ากล้อง")
    print("กด 'q' เพื่อออก, 'g' เพื่อ toggle mask visualization")

    show_mask = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]

        # ---- skin mask (HSV) + denoise ----
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 15, 60])
        upper_skin = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # denoise
        mask = cv2.GaussianBlur(mask, (7,7), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # optional: show mask
        if show_mask:
            cv2.imshow("mask", mask)

        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tips = []

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > MIN_AREA:
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

                # wrist approx (กลางล่างของ bbox)
                wrist = (x + w//2, y + h)

                # hull (points and indices)
                hull_pts = cv2.convexHull(largest, returnPoints=True)[:,0,:]
                hull_idx = cv2.convexHull(largest, returnPoints=False)[:,0]

                # convexity defects (ใช้ indices)
                defects = None
                if len(hull_idx) > 3:
                    defects = cv2.convexityDefects(largest, hull_idx)

                finger_gaps = []
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s_idx, e_idx, f_idx, _ = defects[i,0]
                        start = tuple(largest[s_idx][0])
                        end = tuple(largest[e_idx][0])
                        far = tuple(largest[f_idx][0])

                        # compute geometric depth (distance from far to line start-end)
                        depth = point_line_distance(start, end, far)

                        # angle at far
                        ang = angle_between(start, far, end)

                        # filter small defects by relative depth and angle
                        if ang < DEFECT_ANGLE_TH and depth > (h * DEFECT_DEPTH_RATIO):
                            finger_gaps.append((start, end, far, depth, ang))

                    # collect tip candidates from defect starts/ends
                    for s,e,f,_,_ in finger_gaps:
                        tips.append(s)
                        tips.append(e)

                # fallback: if no defects, use hull points that are above wrist
                if not tips:
                    cand = [tuple(p) for p in hull_pts if p[1] < wrist[1] - 30]
                    tips = cand

                # filter duplicates/close points
                tips = filter_close_points(tips, MIN_DIST)
                tips = sorted(tips, key=lambda p: p[0])  # left->right

                # smoothing: keep history and compute average positions
                tips_history.append(tips)
                # average across history (for variable number of tips, we'll average per-index)
                avg_tips = []
                if tips_history:
                    # build transpose-like lists by index
                    max_len = max(len(t) for t in tips_history)
                    for idx in range(max_len):
                        pts_at_idx = []
                        for t in tips_history:
                            if len(t) > idx:
                                pts_at_idx.append(t[idx])
                        if pts_at_idx:
                            mean_x = int(np.mean([p[0] for p in pts_at_idx]))
                            mean_y = int(np.mean([p[1] for p in pts_at_idx]))
                            avg_tips.append((mean_x, mean_y))
                else:
                    avg_tips = tips

                # draw skeleton for each tip (use avg_tips)
                unique_avg_tips = filter_close_points(avg_tips, MIN_DIST)
                for tip in unique_avg_tips:
                    mcp, pip, dip, tip_point = get_landmarks(wrist, tip)
                    # draw joints
                    for point, c in zip([mcp, pip, dip, tip_point],
                                        [(0,255,255), (0,165,255), (0,100,255), (0,0,255)]):
                        cv2.circle(frame, point, 5, c, -1)
                    # draw bones
                    cv2.line(frame, wrist, mcp, (255,255,255), 2)
                    cv2.line(frame, mcp, pip, (255,255,255), 2)
                    cv2.line(frame, pip, dip, (255,255,255), 2)
                    cv2.line(frame, dip, tip_point, (255,255,255), 2)

                # count fingers (approx): number of unique avg tips
                finger_count = len(unique_avg_tips)
                status = "HAND OPEN" if finger_count >= 3 else "HAND CLOSED"
                color = (0,255,0) if finger_count >= 3 else (0,0,255)
                cv2.putText(frame, f"{status} ({finger_count})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # optional: draw defects for visualization
                for s,e,f,dpt,ang in finger_gaps:
                    cv2.circle(frame, s, 4, (255,0,0), -1)
                    cv2.circle(frame, e, 4, (255,0,0), -1)
                    cv2.circle(frame, f, 4, (0,0,255), -1)
                    cv2.line(frame, s, f, (200,200,0), 1)
                    cv2.line(frame, f, e, (200,200,0), 1)

        cv2.imshow('Hand Detection (no MediaPipe) + Skeleton', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('g'):
            show_mask = not show_mask

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    improved_hand_detection()
