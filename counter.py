from collections import defaultdict, deque
import cv2
import numpy as np
import os
from datetime import datetime
from utils import draw_overlay


def signed_distance_to_line(px, py, x1, y1, x2, y2):
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def point_in_segment_bbox(px, py, x1, y1, x2, y2, margin=40):
    minx = min(x1, x2) - margin
    maxx = max(x1, x2) + margin
    miny = min(y1, y2) - margin
    maxy = max(y1, y2) + margin
    return (minx <= px <= maxx) and (miny <= py <= maxy)


class FootfallCounter:
    def __init__(self, tracker, line_endpoints=None):
        self.tracker = tracker
        self.line_endpoints = line_endpoints
        self.hist = defaultdict(lambda: deque(maxlen=16))
        self.count_in = 0
        self.count_out = 0

        # Create folders
        os.makedirs("logs", exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)

        # Log file path
        self.log_path = "logs/footfall_log.csv"

        # Write header if log file not exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("timestamp,track_id,event,total_in,total_out\n")

    @staticmethod
    def centroid(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def set_line(self, endpoints):
        self.line_endpoints = endpoints

    def log_event(self, track_id, event):
        """Write entry to log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(f"{timestamp},{track_id},{event},{self.count_in},{self.count_out}\n")

    def save_screenshot(self, frame, bbox, track_id):
        """Save cropped person image"""
        x1, y1, x2, y2 = [int(v) for v in bbox]

        crop = frame[max(0, y1):max(1, y2),
                     max(0, x1):max(1, x2)]

        if crop.size == 0:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/person_{track_id}_{timestamp}.jpg"
        cv2.imwrite(filename, crop)

    def process(self, frame, detections):
        h, w = frame.shape[:2]

        # Draw user-defined line
        if self.line_endpoints is not None:
            x1, y1, x2, y2 = self.line_endpoints
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        tracked = self.tracker.update(detections)

        for bbox, tid in tracked:
            x1, y1, x2, y2 = bbox.astype(int)
            cx, cy = self.centroid(bbox)
            self.hist[tid].append((cx, cy))

            # Counting logic
            if self.line_endpoints is not None:
                path = list(self.hist[tid])
                if len(path) >= 2:
                    px, py = path[-2]
                    cx2, cy2 = path[-1]

                    lx1, ly1, lx2, ly2 = self.line_endpoints

                    prev_sd = signed_distance_to_line(px, py, lx1, ly1, lx2, ly2)
                    cur_sd = signed_distance_to_line(cx2, cy2, lx1, ly1, lx2, ly2)

                    if prev_sd * cur_sd < 0:
                        if point_in_segment_bbox(cx2, cy2, lx1, ly1, lx2, ly2):

                            # Person entered
                            if prev_sd < 0 and cur_sd > 0:
                                self.count_in += 1
                                self.log_event(tid, "IN")
                                self.save_screenshot(frame, bbox, tid)
                                self.hist[tid].clear()

                            # Person exited
                            elif prev_sd > 0 and cur_sd < 0:
                                self.count_out += 1
                                self.log_event(tid, "OUT")
                                self.hist[tid].clear()

            # Draw tracking visuals
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f'ID:{tid}', (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Display IN/OUT count
        draw_overlay(frame, [f'IN: {self.count_in}', f'OUT: {self.count_out}'])
        return frame
