import cv2
import numpy as np
from ultralytics import YOLO




class Detector:
    def __init__(self, model_name='yolov8n.pt', conf_thres=0.35, device='auto'):
        self.model = YOLO(model_name)
        self.conf_thres = conf_thres
        try:
            self.names = self.model.names
        except Exception:
            self.names = {0: 'person'}
        self.person_ids = [k for k, v in self.names.items() if v == 'person']
        if not self.person_ids:
            self.person_ids = [0]


    def detect(self, frame: np.ndarray):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(img, conf=self.conf_thres, verbose=False)
        if len(results) == 0:
            return np.zeros((0, 6))
        res = results[0]
        try:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            xyxy = np.array(res.boxes.xyxy)
            confs = np.array(res.boxes.conf)
            cls = np.array(res.boxes.cls).astype(int)


        out = []
        for i in range(len(confs)):
            if cls[i] in self.person_ids:
                x1, y1, x2, y2 = xyxy[i]
                out.append([x1, y1, x2, y2, float(confs[i]), int(cls[i])])
        if len(out) == 0:
            return np.zeros((0, 6))
        return np.array(out)