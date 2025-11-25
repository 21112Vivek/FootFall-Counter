import numpy as np
import cv2




def iou(boxA, boxB):
    # box: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = boxAArea + boxBArea - inter
    if denom <= 0:
        return 0.0
    return inter / denom




def draw_overlay(frame, text_lines, topright=True, box_w=220, alpha=0.6):
    h, w = frame.shape[:2]
    pad = 10
    if topright:
        x0 = w - box_w - pad
        y0 = pad
    else:
        x0 = pad
        y0 = pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + 20 + 25 * len(text_lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (x0 + 10, y0 + 20 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)