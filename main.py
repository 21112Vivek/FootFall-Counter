

import argparse
import time
import cv2
import numpy as np
from detector import Detector
from sort_tracker import Sort
from counter import FootfallCounter

# Global to store user-drawn points (two points define the line)
_mouse_points = []
_drawing_preview = False  # for showing first point while we wait for second


def mouse_callback(event, x, y, flags, param):
    global _mouse_points, _drawing_preview
    # left button down: add a point
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_points.append((x, y))
        if len(_mouse_points) >= 2:
            _drawing_preview = False
        else:
            _drawing_preview = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=str, default='0')
    p.add_argument('--model', type=str, default='yolo11n.pt')
    p.add_argument('--line', type=float, default=0.5)  # unused when manual drawing
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--max-age', type=int, default=30)
    p.add_argument('--min-hits', type=int, default=3)
    return p.parse_args()


def main():
    global _mouse_points, _drawing_preview
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source

    detector = Detector(model_name=args.model, conf_thres=args.conf)
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits)
    counter = FootfallCounter(tracker, line_endpoints=None)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open source {src}')

    win = 'Footfall Counter'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_callback)

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # show guide text before the line is set
        guide_lines = []
        if len(_mouse_points) == 0:
            guide_lines = ["Left-click: first point", "Left-click again: second point"]
        elif len(_mouse_points) == 1:
            guide_lines = ["Left-click: set second point", "Press 'c' to cancel"]
        else:
            guide_lines = ["Line set. Press 'c' to clear and redraw"]

        # run detection
        detections = detector.detect(frame)  # Nx6
        dets_for_sort = detections[:, :5] if detections.shape[0] > 0 else np.zeros((0, 5))

        # If user has drawn two points, send that line to counter
        if len(_mouse_points) >= 2:
            x1, y1 = _mouse_points[0]
            x2, y2 = _mouse_points[1]
            counter.set_line((x1, y1, x2, y2))
        else:
            counter.set_line(None)

        # Process frame (tracking + counting); returns annotated frame
        frame = counter.process(frame, dets_for_sort)

        # Draw interactive preview (first point or temporary line while drawing)
        if len(_mouse_points) == 1:
            # draw first point
            px, py = _mouse_points[0]
            cv2.circle(frame, (px, py), 6, (255, 255, 0), -1)
        elif len(_mouse_points) >= 2:
            # draw the definitive line (counter already draws it, but draw thicker preview)
            x1, y1 = _mouse_points[0]
            x2, y2 = _mouse_points[1]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # draw guide overlay
        # append fps after computing it
        now = time.time()
        fps = 1.0 / (now - prev + 1e-6)
        prev = now
        guide_lines.append(f'FPS: {int(fps)}')
        draw_overlay = None  # keep util name unused; we'll just call utils.draw_overlay from counter

        # show frame
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # clear drawn points and reset line
            _mouse_points = []
            counter.set_line(None)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
