import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
from collections import deque
from hybrid_detector import HybridPPEDetector  # Must have this file & class!

# -----------------------------
# PERSON TRACKER (IOU-based)
# -----------------------------
class PersonTracker:
    def __init__(self, iou_threshold=0.35, max_lost=10):
        self.next_id = 1
        self.tracks = {}
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    @staticmethod
    def _iou(bb1, bb2):
        x1, y1, x2, y2 = bb1
        x1p, y1p, x2p, y2p = bb2
        xx1 = max(x1, x1p)
        yy1 = max(y1, y1p)
        xx2 = min(x2, x2p)
        yy2 = min(y2, y2p)
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter = w * h
        area1 = max(0, (x2 - x1)) * max(0, (y2 - y1))
        area2 = max(0, (x2p - x1p)) * max(0, (y2p - y1p))
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def update(self, detections):
        updated_ids = []
        for det in detections:
            best_id = None
            best_iou = 0.0
            for tid, tdata in self.tracks.items():
                score = self._iou(tdata['bbox'], det)
                if score > best_iou:
                    best_iou = score
                    best_id = tid
            if best_iou >= self.iou_threshold and best_id is not None:
                self.tracks[best_id]['bbox'] = tuple(det)
                self.tracks[best_id]['lost'] = 0
                self.tracks[best_id]['last_seen'] = time.time()
                updated_ids.append(best_id)
            else:
                tid = self.next_id
                self.tracks[tid] = {
                    'bbox': tuple(det),
                    'lost': 0,
                    'ppe': {},
                    'last_seen': time.time()
                }
                updated_ids.append(tid)
                self.next_id += 1
        for tid in list(self.tracks.keys()):
            if tid not in updated_ids:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    del self.tracks[tid]
        return self.tracks

    def assign_ppe(self, detections, ppe_list):
        if not detections or not ppe_list:
            return
        for tid, tdata in self.tracks.items():
            best_idx = None
            best_iou = 0.0
            for idx, det in enumerate(detections):
                score = self._iou(tdata['bbox'], det)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx is not None and best_iou >= self.iou_threshold:
                if best_idx < len(ppe_list):
                    self.tracks[tid]['ppe'] = ppe_list[best_idx].copy()
                    self.tracks[tid]['last_seen'] = time.time()

def draw_person_annotations(frame, tid, bbox, ppe, equipment_names):
    x1, y1, x2, y2 = map(int, bbox)
    required_keys = list(equipment_names.keys())
    all_present = all(ppe.get(k, False) for k in required_keys) if ppe else False
    color = (0, 255, 0) if all_present else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, f"Person {tid}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y_offset = y2 + 20
    for k, label in equipment_names.items():
        present = ppe.get(k, False) if ppe else False
        txt = f"Worn {label}" if present else f"Missing {label}"
        cv2.putText(frame, txt, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if present else (0, 0, 255), 2)
        y_offset += 18
    return frame

# -------------- Main UI Logic --------------
class ProfessionalDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PPE Detection System")
        self.cap = None
        self.tracker = PersonTracker()
        self.equipment_names = {
            "helmet": "Helmet",
            "vest": "Safety Vest",
            "gloves": "Gloves",
            "safety_glasses": "Safety Glasses"
        }
        self.detector = HybridPPEDetector()  # Must be implemented in hybrid_detector.py
        self.last_popup_time = 0
        self.popup_cooldown = 1.5
        self.setup_ui()
        self.running = False

    def setup_ui(self):
        self.video_label = tk.Label(self.root)
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10)
        rightframe = tk.Frame(self.root, bg="#1e1e1e", width=420)
        rightframe.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        rightframe.pack_propagate(False)
        overallframe = tk.LabelFrame(rightframe, text="Overall Status", font=("Arial", 14, "bold"), bg="#2d2d2d", fg="#ffffff", padx=15, pady=15)
        overallframe.pack(fill=tk.X, padx=10)
        self.overallstatuslabel = tk.Label(overallframe, text="Status: Not Started", font=("Arial", 16, "bold"), bg="#2d2d2d", fg="#888888")
        self.overallstatuslabel.pack(pady=10)

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            self.stop_detection()
            return
        self.running = True
        threading.Thread(target=self.run_detection_loop).start()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def run_detection_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue
            detections, ppe_list = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
            self.tracker.assign_ppe(detections, ppe_list)
            for tid, tdata in tracks.items():
                draw_person_annotations(frame, tid, tdata['bbox'], tdata.get('ppe', {}), self.equipment_names)
            self.root.after(0, self.update_video_display, frame)
            time.sleep(0.03)

    def update_video_display(self, frame):
        display_height = 600
        aspect_ratio = frame.shape[1] / frame.shape[0]
        display_width = int(display_height * aspect_ratio)
        frame_resized = cv2.resize(frame, (display_width, display_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.config(image=photo)
        self.video_label.image = photo

def main():
    root = tk.Tk()
    app = ProfessionalDetectionUI(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_detection)
    app.start_detection()
    root.mainloop()

if __name__ == "__main__":
    main()
