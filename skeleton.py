# skeleton/vis.py
import cv2

def draw_overlay(img, bbox, landmarks, states):
    x,y,w,h = bbox
    cv2.rectangle(img, (x,y), (x+w, y+h), (200,200,200), 1)
    # draw landmarks & lines (use MyMediapipe.draw_landmarks if available)
    from MyMediapipe import draw_landmarks
    draw_landmarks(img, landmarks)

    # draw states text
    for i,s in enumerate(states):
        a,b = s['pair']
        txt = f"{a}-{b} {'TOUCH' if s['touching'] else 'OPEN'} {s['norm_dist']:.2f}"
        cv2.putText(img, txt, (10, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255) if s['touching'] else (150,150,150), 2)
