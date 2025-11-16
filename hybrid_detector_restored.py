"""
Hybrid Safety Kit Detection System (restored copy)
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import time
from collections import deque
import threading

# UI imports with error handling
try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available. UI features will be disabled.")

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Install with: pip install Pillow")


class HybridPPEDetector:
    def __init__(self):
        self.person_model = YOLO('yolov8n.pt')
        self.enhancer = self._create_enhancer()
        self.detection_history = {
            'helmet': deque(maxlen=10),
            'vest': deque(maxlen=10),
            'gloves': deque(maxlen=10),
            'safety_glasses': deque(maxlen=10)
        }
        self.last_vest_mask = None
        self.last_bright_mask = None

    def _create_enhancer(self):
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def enhance_image(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.enhancer.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return enhanced

    def detect_person(self, frame: np.ndarray) -> List[Tuple]:
        results = self.person_model(frame, classes=[0], conf=0.5, verbose=False)
        persons = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                persons.append((bbox, conf))
        return persons

    # Other detection methods (helmet, vest, glasses, gloves)...
    def detect_helmet(self, frame, person_bbox):
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        head_region = frame[y1:y1 + int(h * 0.3), x1:x2]
        if head_region.size == 0:
            return False
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        if std_dev > 30:
            return False
        return True

    def detect_vest(self, frame, person_bbox):
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        torso_region = frame[y1 + int(h * 0.2):y1 + int(h * 0.7), x1:x2]
        if torso_region.size == 0:
            return False
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        h_roi, w_roi = hsv.shape[:2]
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_ratio = np.sum(yellow_mask > 0) / (h_roi * w_roi)
        if yellow_ratio > 0.25:
            self.last_vest_mask = (yellow_mask, (x1, y1 + int(h * 0.2)))
            return True
        return False

    def detect_safety_glasses(self, frame, person_bbox):
        return False

    def detect_gloves(self, frame, person_bbox):
        return False

    def check_temporal_consistency(self, equipment: str, detected: bool) -> bool:
        self.detection_history[equipment].append(detected)
        if len(self.detection_history[equipment]) < 5:
            return detected
        recent = list(self.detection_history[equipment])[-5:]
        return sum(recent) >= 3

    def detect(self, frame: np.ndarray) -> Dict:
        enhanced = self.enhance_image(frame)
        persons = self.detect_person(enhanced)
        detected_equipment = {'helmet': False, 'vest': False, 'gloves': False, 'safety_glasses': False, 'person': len(persons) > 0}
        person_bboxes = []
        for bbox, conf in persons:
            person_bboxes.append(bbox)
            helmet = self.detect_helmet(enhanced, bbox)
            vest = self.detect_vest(enhanced, bbox)
            gloves = self.detect_gloves(enhanced, bbox)
            glasses = self.detect_safety_glasses(enhanced, bbox)
            if self.check_temporal_consistency('helmet', helmet):
                detected_equipment['helmet'] = True
            if self.check_temporal_consistency('vest', vest):
                detected_equipment['vest'] = True
            if self.check_temporal_consistency('gloves', gloves):
                detected_equipment['gloves'] = True
            if self.check_temporal_consistency('safety_glasses', glasses):
                detected_equipment['safety_glasses'] = True
        return {'equipment': detected_equipment, 'detections': [], 'person_bboxes': person_bboxes}


class PersonTracker:
    def __init__(self):
        self.persons = {}
        self.next_id = 0
        self.max_disappear_time = 2.0

    def _center(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _dist(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def update_persons(self, bboxes, equipment_list):
        now = time.time()
        if not self.persons:
            for i, bbox in enumerate(bboxes):
                self.persons[self.next_id] = {'bbox': bbox, 'last_seen': now, 'equipment': equipment_list[i] if i < len(equipment_list) else {}}
                self.next_id += 1
            return
        centers = [self._center(b) for b in bboxes]
        used = set()
        for i, c in enumerate(centers):
            best_id = None
            best_d = 1e9
            for pid, pdata in self.persons.items():
                if pid in used:
                    continue
                d = self._dist(c, self._center(pdata['bbox']))
                if d < best_d:
                    best_d = d
                    best_id = pid
            if best_id is not None and best_d < 150:
                self.persons[best_id]['bbox'] = bboxes[i]
                self.persons[best_id]['last_seen'] = now
                if i < len(equipment_list):
                    self.persons[best_id]['equipment'] = equipment_list[i]
                used.add(best_id)
            else:
                self.persons[self.next_id] = {'bbox': bboxes[i], 'last_seen': now, 'equipment': equipment_list[i] if i < len(equipment_list) else {}}
                self.next_id += 1
        stale = [pid for pid, p in self.persons.items() if now - p['last_seen'] > self.max_disappear_time]
        for pid in stale:
            del self.persons[pid]

    def get_all_persons(self):
        return self.persons


class HybridDetectionApp:
    def __init__(self, root=None):
        if not TKINTER_AVAILABLE:
            raise ImportError("tkinter required")
        if not PIL_AVAILABLE:
            raise ImportError("Pillow required")
        if root is None:
            root = tk.Tk()
        self.root = root
        self.root.title("Safety Kit Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')

        self.detector = HybridPPEDetector()
        self.tracker = PersonTracker()
        self.cap = None
        self.running = False
        self.setup_ui()

    def setup_ui(self):
        pass


if __name__ == '__main__':
    print('Restored copy written to hybrid_detector_restored.py')
