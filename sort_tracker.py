import numpy as np
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
from typing import List, Tuple
from utils import iou


from scipy.optimize import linear_sum_assignment




@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    kf: KalmanFilter
    age: int = 0
    time_since_update: int = 0
    hits: int = 0
    hit_streak: int = 0


    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        x, y, _, _, w, h = self.kf.x.flatten()
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        self.bbox = np.array([x1, y1, x2, y2])
        return self.bbox


    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        z = np.array([cx, cy, w, h]).reshape((4, 1))
        self.kf.update(z)
        self.bbox = np.array([x1, y1, x2, y2])
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1




class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self._next_id = 1


    def _create_kf(self, bbox):
        kf = KalmanFilter(dim_x=6, dim_z=4)
        dt = 1.
        kf.F = np.array([
        [1, 0, dt, 0, 0, 0],
        [0, 1, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        ])
        kf.H = np.zeros((4, 6))
        kf.H[0, 0] = 1
        kf.H[1, 1] = 1
        kf.H[2, 4] = 1
        kf.H[3, 5] = 1
        kf.P *= 10.
        kf.R *= 1.
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        kf.x = np.array([[cx], [cy], [0.], [0.], [w], [h]])
        return kf

    def update(self, detections: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        # detections: Nx5 array
        for tr in self.tracks:
                tr.predict()


        N = len(self.tracks)
        M = len(detections)


        if N == 0:
            for i in range(M):
                kf = self._create_kf(detections[i, :4])
                tr = Track(self._next_id, detections[i, :4].copy(), kf)
                self._next_id += 1
                self.tracks.append(tr)
            return [(t.bbox, t.track_id) for t in self.tracks]


        iou_matrix = np.zeros((N, M), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            for j in range(M):
                iou_matrix[i, j] = iou(tr.bbox, detections[j, :4])


        cost = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)


        assigned_tracks = set()
        assigned_dets = set()
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                self.tracks[r].update(detections[c, :4])
                assigned_tracks.add(r)
                assigned_dets.add(c)


        for j in range(M):
            if j not in assigned_dets:
                kf = self._create_kf(detections[j, :4])
                tr = Track(self._next_id, detections[j, :4].copy(), kf)
                self._next_id += 1
                self.tracks.append(tr)


        for i in reversed(range(len(self.tracks))):
            tr = self.tracks[i]
            if tr.time_since_update > self.max_age:
                self.tracks.pop(i)


        results = []
        for tr in self.tracks:
            if tr.hits >= self.min_hits or tr.time_since_update == 0:
                results.append((tr.bbox.copy(), tr.track_id))
        return results