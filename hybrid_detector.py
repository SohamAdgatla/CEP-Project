import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
from collections import deque
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading

# ----------------------------------------------------------------------------
# HybridPPEDetector
# ----------------------------------------------------------------------------

class HybridPPEDetector:
    """Hybrid detector using CV techniques + YOLO person detection"""
    def __init__(self):
        self.person_model = YOLO('yolov8n.pt')
        self.enhancer = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.detection_history = {
            'helmet': deque(maxlen=10),
            'vest': deque(maxlen=10),
            'gloves': deque(maxlen=10),
            'safety_glasses': deque(maxlen=10)
        }
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
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                persons.append((bbox, conf))
        return persons
    def detect_helmet(self, frame: np.ndarray, person_bbox: Tuple) -> bool:
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        head_region = frame[y1:y1 + int(h * 0.3), x1:x2]
        if head_region.size == 0:
            return False
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = gray.shape
        std_dev = np.std(gray)
        if std_dev > 30:
            return False
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h_roi * w_roi)
        if edge_density > 0.2:
            return False
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.25:
                    return False
        mean_brightness = np.mean(gray)
        if mean_brightness < 25 or mean_brightness > 220:
            return False
        return True
    def detect_vest(self, frame: np.ndarray, person_bbox: Tuple) -> bool:
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        torso_region = frame[y1 + int(h * 0.2):y1 + int(h * 0.7), x1:x2]
        if torso_region.size == 0:
            return False
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        h_roi, w_roi = hsv.shape[:2]
        yellow_mask = cv2.inRange(hsv, np.array([20,100,100]), np.array([30,255,255]))
        orange_mask = cv2.inRange(hsv, np.array([10,100,100]), np.array([20,255,255]))
        yellow_ratio = np.sum(yellow_mask > 0) / (h_roi * w_roi)
        orange_ratio = np.sum(orange_mask > 0) / (h_roi * w_roi)
        v_channel = hsv[:,:,2]
        brightness = np.mean(v_channel)
        if yellow_ratio > 0.25 or orange_ratio > 0.25:
            return True
        if brightness > 150:
            bright_areas = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)[1]
            bright_ratio = np.sum(bright_areas > 0) / (h_roi * w_roi)
            if bright_ratio > 0.15:
                return True
        return False
    def detect_safety_glasses(self, frame: np.ndarray, person_bbox: Tuple) -> bool:
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        w = x2 - x1
        face_region = frame[y1 + int(h*0.15):y1 + int(h*0.4), x1 + int(w*0.2):x2 - int(w*0.2)]
        if face_region.size == 0:
            return False
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = gray.shape
        edges = cv2.Canny(gray, 50, 150)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_roi//2, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_ratio = np.sum(horizontal_lines > 0) / (h_roi * w_roi)
        if horizontal_ratio < 0.03:
            return False
        edge_density = np.sum(edges > 0) / (h_roi * w_roi)
        if edge_density < 0.06 or edge_density > 0.3:
            return False
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            return False
        left_half = edges[:, :w_roi//2]
        right_half = edges[:, w_roi//2:]
        right_flipped = cv2.flip(right_half, 1)
        if left_half.shape == right_flipped.shape:
            similarity = cv2.matchTemplate(left_half, right_flipped, cv2.TM_CCOEFF_NORMED)[0][0]
            if similarity < 0.25:
                return False
        return True
    def detect_gloves(self, frame: np.ndarray, person_bbox: Tuple) -> bool:
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        w = x2 - x1
        hand_y_start = y1 + int(h * 0.6)
        hand_y_end = y2
        hand_width = int(w * 0.15)
        left_hand = frame[hand_y_start:hand_y_end, x1:x1 + hand_width]
        right_hand = frame[hand_y_start:hand_y_end, x2 - hand_width:x2]
        for hand_region in [left_hand, right_hand]:
            if hand_region.size == 0:
                continue
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / gray.size
            if 0.08 < edge_density < 0.35:
                mean_color = np.mean(gray)
                if 40 < mean_color < 180:
                    return True
        return False
    def check_temporal_consistency(self, equipment: str, detected: bool) -> bool:
        self.detection_history[equipment].append(detected)
        if len(self.detection_history[equipment]) < 5:
            return detected
        recent = list(self.detection_history[equipment])[-5:]
        true_count = sum(recent)
        return true_count >= 3
    def detect(self, frame: np.ndarray) -> Dict:
        enhanced = self.enhance_image(frame)
        persons = self.detect_person(enhanced)
        detected_equipment = {
            'helmet': False, 'vest': False, 'gloves': False, 'safety_glasses': False, 'person': bool(persons)
        }
        if not persons:
            return {'equipment': detected_equipment, 'detections': [], 'person_bboxes': []}
        for person_bbox, _ in persons:
            helmet = self.detect_helmet(enhanced, person_bbox)
            vest = self.detect_vest(enhanced, person_bbox)
            glasses = self.detect_safety_glasses(enhanced, person_bbox)
            gloves = self.detect_gloves(enhanced, person_bbox)
            if self.check_temporal_consistency('helmet', helmet): detected_equipment['helmet'] = True
            if self.check_temporal_consistency('vest', vest): detected_equipment['vest'] = True
            if self.check_temporal_consistency('safety_glasses', glasses): detected_equipment['safety_glasses'] = True
            if self.check_temporal_consistency('gloves', gloves): detected_equipment['gloves'] = True
        return {
            'equipment': detected_equipment,
            'detections': [],
            'person_bboxes': [bbox for bbox, _ in persons]
        }

# ----------------------------------------------------------------------------
# PersonTracker
# ----------------------------------------------------------------------------

class PersonTracker:
    """Track equipment status per person"""
    def __init__(self):
        self.persons = {}  # {person_id: {equipment: dict, bbox: tuple, last_seen: time}}
        self.next_id = 0
    def update_persons(self, person_bboxes, person_equipment_list):
        current_time = time.time()
        matched = set()
        for person_id, person_data in list(self.persons.items()):
            if current_time - person_data['last_seen'] > 2.0:
                del self.persons[person_id]
                continue
            matched_id = None
            min_dist = float('inf')
            for idx, bbox in enumerate(person_bboxes):
                if idx in matched:
                    continue
                dist = self._calculate_distance(person_data['bbox'], bbox)
                if dist < min_dist and dist < 100:
                    min_dist = dist
                    matched_id = idx
            if matched_id is not None:
                matched.add(matched_id)
                self.persons[person_id]['bbox'] = person_bboxes[matched_id]
                if matched_id < len(person_equipment_list):
                    self.persons[person_id]['equipment'] = person_equipment_list[matched_id].copy()
                self.persons[person_id]['last_seen'] = current_time
        for idx, bbox in enumerate(person_bboxes):
            if idx not in matched:
                equipment = person_equipment_list[idx] if idx < len(person_equipment_list) else {}
                self.persons[self.next_id] = {
                    'equipment': equipment.copy(), 'bbox': bbox, 'last_seen': current_time
                }
                self.next_id += 1
    def _calculate_distance(self, bbox1, bbox2):
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        return np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    def get_all_persons(self):
        return self.persons.copy()

# ----------------------------------------------------------------------------
# HybridDetectionApp
# ----------------------------------------------------------------------------

class HybridDetectionApp:
    """Professional UI Application using hybrid detection with per-person tracking."""
    def __init__(self, root=None):
        self.root = root if root is not None else tk.Tk()
        self.root.title("Safety Kit Detection System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        self.detector = HybridPPEDetector()
        self.tracker = PersonTracker()
        self.cap = None
        self.running = False
        self.last_popup_time = 0
        self.popup_cooldown = 5
        self.equipment_names = {
            'helmet': 'Helmet',
            'vest': 'Safety Vest',
            'gloves': 'Gloves',
            'safety_glasses': 'Safety Glasses'
        }
        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 20, 'bold'), background='#1e1e1e', foreground='#00ff88')
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        title_frame = tk.Frame(main_frame, bg='#1e1e1e', height=60)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        title_frame.pack_propagate(False)
        tk.Label(title_frame, text="🛡️ Safety Kit Detection System",
            font=('Arial', 24, 'bold'), bg='#1e1e1e', fg='#00ff88'
        ).pack(side=tk.LEFT, padx=20, pady=15)
        self.status_indicator = tk.Label(title_frame, text="● Ready", font=('Arial',12), bg='#1e1e1e', fg='#00ff88')
        self.status_indicator.pack(side=tk.RIGHT, padx=20, pady=15)
        content_frame = tk.Frame(main_frame, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True)
        left_frame = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        video_label_frame = tk.LabelFrame(left_frame, text="Camera Feed",
            font=('Arial',12,'bold'), bg='#2d2d2d', fg='#fff', padx=10, pady=10)
        video_label_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.video_label = tk.Label(video_label_frame, text="Camera not started", bg='#1a1a1a', fg='#888', font=('Arial',14))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        button_frame = tk.Frame(left_frame, bg='#2d2d2d')
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        self.start_btn = tk.Button(button_frame, text="▶ Start Detection", command=self.start_detection,
            bg='#00ff88', fg='#000', font=('Arial',12,'bold'), relief=tk.FLAT, padx=20, pady=10, cursor='hand2')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(button_frame, text="⏹ Stop Detection", command=self.stop_detection,
            bg='#ff4444', fg='#fff', font=('Arial',12,'bold'), relief=tk.FLAT, padx=20, pady=10, cursor='hand2', state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        right_frame = tk.Frame(content_frame, bg='#1e1e1e', width=450)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10,0))
        right_frame.pack_propagate(False)
        overall_frame = tk.LabelFrame(right_frame, text="Overall Status",
            font=('Arial',14,'bold'), bg='#2d2d2d', fg='#fff', padx=15, pady=15)
        overall_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        self.overall_status_label = tk.Label(overall_frame, text="Status: Not Started",
            font=('Arial',16,'bold'), bg='#2d2d2d', fg='#888')
        self.overall_status_label.pack(pady=10)
        # --- SMALL PERSONS PANEL
        self.persons_scrollable_frame = self._setup_scrollable_panel(
            right_frame, "Persons Detected", fg='#1e90ff', height=60
        )
        # --- WEARED ITEMS PANEL
        self.worn_scrollable_frame = self._setup_scrollable_panel(
            right_frame, "✓ Weared Items", fg='#00cc99')
        # --- UNWEARED ITEMS PANEL
        self.unworn_scrollable_frame = self._setup_scrollable_panel(
            right_frame, "✗ Unweared Items", fg='#ff944d')
        # --- POPUP AREA
        self.popup_frame = tk.Frame(self.root, bg="#eafaf1", bd=2, relief=tk.RIDGE)
        self.popup_frame.place(relx=1.0, rely=1.0, x=-12, y=-12, anchor='se')
        self.popup_msg = tk.Label(self.popup_frame, text="", font=('Arial',14,'bold'), bg="#eafaf1", fg="#357a38", wraplength=350, justify="left")
        self.popup_msg.pack(padx=12, pady=8)
        self.last_popup_status = None

    def _setup_scrollable_panel(self, frame, text, fg='#00ff88', height=120):
        panel = tk.LabelFrame(frame, text=text, font=('Arial',14,'bold'), bg='#2d2d2d', fg=fg, padx=15, pady=5, height=height)
        panel.pack(fill=tk.X, padx=10, pady=(0,8))
        panel.pack_propagate(False)  # Make panel fixed height
        canvas = tk.Canvas(panel, bg='#2d2d2d', highlightthickness=0, height=height)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#2d2d2d')
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return scrollable_frame

    # -- Detection loop and UI methods --
    def start_detection(self):
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_indicator.config(text="● Detecting", fg='#00ff88')
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            self.stop_detection()
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        self.cap.set(cv2.CAP_PROP_FPS,30)
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_indicator.config(text="● Stopped", fg='#ff4444')
        self.overall_status_label.config(text="Status: Stopped", fg='#888')

    def detection_loop(self):
        frame_count = 0
        results = {'equipment': {}, 'person_bboxes': [], 'detections': []}
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            if frame_count % 2 == 0:
                results = self.detector.detect(frame)
                if results.get('person_bboxes'):
                    person_equipment_list = []
                    enhanced = self.detector.enhance_image(frame)
                    for bbox in results['person_bboxes']:
                        person_eq = {
                            'helmet': self.detector.detect_helmet(enhanced, bbox),
                            'vest': self.detector.detect_vest(enhanced, bbox),
                            'safety_glasses': self.detector.detect_safety_glasses(enhanced, bbox),
                            'gloves': self.detector.detect_gloves(enhanced, bbox)
                        }
                        person_eq['helmet'] = self.detector.check_temporal_consistency('helmet', person_eq['helmet'])
                        person_eq['vest'] = self.detector.check_temporal_consistency('vest', person_eq['vest'])
                        person_eq['safety_glasses'] = self.detector.check_temporal_consistency('safety_glasses', person_eq['safety_glasses'])
                        person_eq['gloves'] = self.detector.check_temporal_consistency('gloves', person_eq['gloves'])
                        person_equipment_list.append(person_eq)
                    self.tracker.update_persons(results['person_bboxes'], person_equipment_list)
                self.process_detection_results(results)
            annotated_frame = self.draw_detections(frame_copy, results)
            self.update_video_display(annotated_frame)
            self.root.after(0, self.update_ui_display, results)
            frame_count += 1
            time.sleep(0.03)
        if self.cap:
            self.cap.release()

    def draw_detections(self, frame, results):
        persons = self.tracker.get_all_persons()
        for person_id, person_data in persons.items():
            bbox = person_data['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            eq = person_data['equipment']
            all_weared = all([eq.get(k,False) for k in self.equipment_names])
            color = (0,255,0) if all_weared else (0,0,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
            label = f"P{person_id+1}"
            cv2.putText(frame, label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    def process_detection_results(self, results):
        # No OS popup, handled via the live popup panel.
        pass

    def update_video_display(self, frame):
        if frame is None or frame.size == 0:
            return
        display_height = 600
        aspect_ratio = frame.shape[1] / frame.shape[0]
        display_width = int(display_height * aspect_ratio)
        frame_resized = cv2.resize(frame, (display_width, display_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.root.after(0, lambda: self.video_label.config(image=photo))
        self.root.after(0, lambda: setattr(self, 'current_photo', photo))

    def update_ui_display(self, results):
        equipment = results.get('equipment', {})
        persons_data = self.tracker.get_all_persons()
        all_required = list(self.equipment_names.keys())
        all_weared = all(equipment.get(eq,False) for eq in all_required) if equipment.get('person',False) else False
        if not equipment.get('person',False):
            status_text, status_color = "No person detected", '#888'
        elif all_weared:
            status_text, status_color = "✓ All Equipment Weared", '#00ff88'
        else:
            status_text, status_color = "✗ Missing Equipment", '#ff4444'
        self.overall_status_label.config(text=f"Status: {status_text}", fg=status_color)
        # Clear panels
        for f in [self.persons_scrollable_frame, self.worn_scrollable_frame, self.unworn_scrollable_frame]:
            for w in f.winfo_children():
                w.destroy()
        for pid, pdata in persons_data.items():
            tk.Label(self.persons_scrollable_frame, text=f"P{pid+1}", font=('Arial',11,'bold'),
                bg='#3a3a3a', fg='#fff', width=8, anchor=tk.CENTER).pack(side=tk.LEFT, padx=6, pady=9)
        all_weared_items, all_unweared_items = set(), set()
        for pid, pdata in persons_data.items():
            eq = pdata['equipment']
            for k, v in self.equipment_names.items():
                if eq.get(k,False):
                    all_weared_items.add(f"Person {pid+1}: {v}")
                else:
                    all_unweared_items.add(f"Person {pid+1}: {v}")
        if all_weared_items:
            for item in sorted(all_weared_items):
                tk.Label(self.worn_scrollable_frame, text=f"✓ {item}",
                    font=('Arial',10), bg='#2d2d2d', fg='#00cc99', anchor=tk.W, padx=5, pady=3
                ).pack(fill=tk.X, pady=2)
        else:
            tk.Label(self.worn_scrollable_frame, text="No items weared",
                font=('Arial',10), bg='#2d2d2d', fg='#a7a7a7', anchor=tk.W, padx=5, pady=5
            ).pack(fill=tk.X, pady=2)
        if all_unweared_items:
            for item in sorted(all_unweared_items):
                tk.Label(self.unworn_scrollable_frame, text=f"✗ {item}",
                    font=('Arial',10), bg='#2d2d2d', fg='#ff944d', anchor=tk.W, padx=5, pady=3
                ).pack(fill=tk.X, pady=2)
        else:
            tk.Label(self.unworn_scrollable_frame, text="All items weared",
                font=('Arial',10), bg='#2d2d2d', fg='#a7a7a7', anchor=tk.W, padx=5, pady=5
            ).pack(fill=tk.X, pady=2)
        # POPUP panel
        all_persons_count = max(1, len(persons_data))
        if all_weared and equipment.get('person',False):
            text = f"All required equipment weared for {all_persons_count} person(s).\n\nYou can go!"
            bg, fg = "#eafaf1", "#38b000"
        elif equipment.get('person',False):
            missing_items_flat = sorted(set(i.split(":")[1].strip() for i in all_unweared_items))
            text = "Please wear the remaining items:\n• " + "\n• ".join(missing_items_flat)
            bg, fg = "#fff7e6", "#ec9a00"
        else:
            text = "No person detected."
            bg, fg = "#f2f2f2", "#888"
        if (text, bg, fg) != self.last_popup_status:
            self.popup_frame.config(bg=bg)
            self.popup_msg.config(text=text, bg=bg, fg=fg)
            self.last_popup_status = (text, bg, fg)

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = HybridDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
