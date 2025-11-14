"""
Hybrid Safety Kit Detection System
Uses computer vision techniques + YOLO for detection without requiring custom training
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import time
from collections import deque

class HybridPPEDetector:
    """Hybrid detector using CV techniques + YOLO person detection"""
    
    def __init__(self):
        # Load YOLO for person detection
        self.person_model = YOLO('yolov8n.pt')
        self.enhancer = self._create_enhancer()
        
        # Detection history for temporal filtering
        self.detection_history = {
            'helmet': deque(maxlen=10),
            'vest': deque(maxlen=10),
            'gloves': deque(maxlen=10),
            'safety_glasses': deque(maxlen=10)
        }
        
    def _create_enhancer(self):
        """Create CLAHE enhancer"""
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    def enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """Enhance image for better detection"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.enhancer.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return enhanced
    
    def detect_person(self, frame: np.ndarray) -> List[Tuple]:
        """Detect person in frame using YOLO"""
        results = self.person_model(frame, classes=[0], conf=0.5, verbose=False)  # class 0 = person
        persons = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                persons.append((bbox, conf))
        return persons
    
    def detect_helmet(self, frame: np.ndarray, person_bbox: Tuple) -> bool:
        """Detect helmet in upper region of person"""
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        w = x2 - x1
        
        # Check upper 30% of person (head region)
        head_region = frame[y1:y1 + int(h * 0.3), x1:x2]
        
        if head_region.size == 0:
            return False
        
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = gray.shape
        
        # Helmet characteristics:
        # 1. More uniform color (less variation than hair)
        std_dev = np.std(gray)
        if std_dev > 30:  # Too much variation = likely hair
            return False
        
        # 2. Hard edges (helmet has defined edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h_roi * w_roi)
        
        # Hair has higher edge density
        if edge_density > 0.2:
            return False
        
        # 3. Shape - helmet is typically more circular
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.25:  # Not circular enough
                    return False
        
        # 4. Color consistency
        mean_brightness = np.mean(gray)
        if mean_brightness < 25 or mean_brightness > 220:  # Too dark or too bright
            return False
        
        return True
    
    def detect_vest(self, frame: np.ndarray, person_bbox: Tuple) -> bool:
        """Detect safety vest using color analysis"""
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        
        # Check torso region (middle 50% of person)
        torso_region = frame[y1 + int(h * 0.2):y1 + int(h * 0.7), x1:x2]
        
        if torso_region.size == 0:
            return False
        
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        h_roi, w_roi = hsv.shape[:2]
        
        # Safety vest colors: yellow and orange
        # Yellow range
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_ratio = np.sum(yellow_mask > 0) / (h_roi * w_roi)
        
        # Orange range
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([20, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        orange_ratio = np.sum(orange_mask > 0) / (h_roi * w_roi)
        
        # Bright/reflective areas (safety vests are often bright)
        v_channel = hsv[:, :, 2]
        brightness = np.mean(v_channel)
        
        # Must have significant yellow/orange OR high brightness
        if yellow_ratio > 0.25 or orange_ratio > 0.25:
            return True
        
        # Check for bright reflective material
        if brightness > 150:
            bright_areas = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)[1]
            bright_ratio = np.sum(bright_areas > 0) / (h_roi * w_roi)
            if bright_ratio > 0.15:
                return True
        
        return False
    
    def detect_safety_glasses(self, frame: np.ndarray, person_bbox: Tuple) -> bool:
        """Detect safety glasses in face region"""
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        w = x2 - x1
        
        # Check upper-middle region (face area)
        face_region = frame[y1 + int(h * 0.15):y1 + int(h * 0.4), 
                           x1 + int(w * 0.2):x2 - int(w * 0.2)]
        
        if face_region.size == 0:
            return False
        
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = gray.shape
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for horizontal lines (glasses frame)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_roi//2, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_ratio = np.sum(horizontal_lines > 0) / (h_roi * w_roi)
        
        if horizontal_ratio < 0.03:  # No clear frame structure
            return False
        
        # Check for lens areas (darker regions with defined edges)
        edge_density = np.sum(edges > 0) / (h_roi * w_roi)
        if edge_density < 0.06 or edge_density > 0.3:
            return False
        
        # Look for two distinct regions (lenses)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return False
        
        # Check symmetry (glasses are typically symmetric)
        left_half = edges[:, :w_roi//2]
        right_half = edges[:, w_roi//2:]
        right_flipped = cv2.flip(right_half, 1)
        
        if left_half.shape == right_flipped.shape:
            similarity = cv2.matchTemplate(left_half, right_flipped, cv2.TM_CCOEFF_NORMED)[0][0]
            if similarity < 0.25:
                return False
        
        return True
    
    def detect_gloves(self, frame: np.ndarray, person_bbox: Tuple) -> bool:
        """Detect gloves in hand regions"""
        x1, y1, x2, y2 = map(int, person_bbox)
        h = y2 - y1
        w = x2 - x1
        
        # Check lower regions (hand areas) - left and right sides
        hand_y_start = y1 + int(h * 0.6)
        hand_y_end = y2
        hand_width = int(w * 0.15)
        
        # Left hand region
        left_hand = frame[hand_y_start:hand_y_end, x1:x1 + hand_width]
        # Right hand region
        right_hand = frame[hand_y_start:hand_y_end, x2 - hand_width:x2]
        
        detected = False
        
        for hand_region in [left_hand, right_hand]:
            if hand_region.size == 0:
                continue
            
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
            
            # Gloves typically have:
            # - Distinct texture (not smooth like skin)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / gray.size
            
            # Moderate texture indicates gloves
            if 0.08 < edge_density < 0.35:
                # Check color - gloves are often distinct from skin
                mean_color = np.mean(gray)
                if 40 < mean_color < 180:  # Not too dark or too bright
                    detected = True
                    break
        
        return detected
    
    def check_temporal_consistency(self, equipment: str, detected: bool) -> bool:
        """Check if detection is consistent over time"""
        self.detection_history[equipment].append(detected)
        
        if len(self.detection_history[equipment]) < 5:
            # Need more history
            return detected
        
        # Check last 5 frames
        recent = list(self.detection_history[equipment])[-5:]
        true_count = sum(recent)
        
        # Must be detected in at least 60% of recent frames
        return true_count >= 3
    
    def detect(self, frame: np.ndarray) -> Dict:
        """Main detection function"""
        enhanced = self.enhance_image(frame)
        
        # Detect persons
        persons = self.detect_person(enhanced)
        
        detected_equipment = {
            'helmet': False,
            'vest': False,
            'gloves': False,
            'safety_glasses': False,
            'person': len(persons) > 0
        }
        
        if not persons:
            return {
                'equipment': detected_equipment,
                'detections': [],
                'person_bboxes': []
            }
        
        # For each person, detect PPE
        for person_bbox, conf in persons:
            # Detect each equipment type
            helmet = self.detect_helmet(enhanced, person_bbox)
            vest = self.detect_vest(enhanced, person_bbox)
            glasses = self.detect_safety_glasses(enhanced, person_bbox)
            gloves = self.detect_gloves(enhanced, person_bbox)
            
            # Check temporal consistency
            if self.check_temporal_consistency('helmet', helmet):
                detected_equipment['helmet'] = True
            if self.check_temporal_consistency('vest', vest):
                detected_equipment['vest'] = True
            if self.check_temporal_consistency('safety_glasses', glasses):
                detected_equipment['safety_glasses'] = True
            if self.check_temporal_consistency('gloves', gloves):
                detected_equipment['gloves'] = True
        
        return {
            'equipment': detected_equipment,
            'detections': [],
            'person_bboxes': [bbox for bbox, _ in persons]
        }

class HybridDetectionApp:
    """Application using hybrid detection"""
    
    def __init__(self):
        self.detector = HybridPPEDetector()
        self.cap = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results"""
        frame_copy = frame.copy()
        
        # Draw person bounding boxes
        for bbox in results['person_bboxes']:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_copy, "PERSON", (x1, y1 - 10),
                       self.font, 0.6, (255, 0, 0), 2)
        
        # Status panel
        equipment = results['equipment']
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 320, 180
        
        # Background
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
        cv2.rectangle(frame_copy, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame_copy, "SAFETY STATUS", (panel_x + 10, panel_y + 30),
                   self.font, 0.8, (255, 255, 255), 2)
        
        # Equipment status
        y_offset = 60
        items = [
            ('Helmet', equipment['helmet']),
            ('Safety Vest', equipment['vest']),
            ('Gloves', equipment['gloves']),
            ('Safety Glasses', equipment['safety_glasses'])
        ]
        
        for item_name, detected in items:
            color = (0, 255, 0) if detected else (0, 0, 255)
            status = "DETECTED" if detected else "MISSING"
            cv2.putText(frame_copy, f"{item_name}: {status}",
                       (panel_x + 10, panel_y + y_offset),
                       self.font, 0.5, color, 2)
            y_offset += 30
        
        # Overall status
        all_detected = all([equipment['helmet'], equipment['vest'],
                           equipment['gloves'], equipment['safety_glasses']])
        overall_color = (0, 255, 0) if all_detected else (0, 0, 255)
        overall_status = "SAFE" if all_detected else "UNSAFE"
        cv2.putText(frame_copy, f"OVERALL: {overall_status}",
                   (panel_x + 10, panel_y + y_offset + 10),
                   self.font, 0.7, overall_color, 2)
        
        return frame_copy
    
    def run(self):
        """Run the application"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("=" * 60)
        print("Hybrid Safety Kit Detection System")
        print("=" * 60)
        print("Uses computer vision techniques for PPE detection")
        print("Press 'q' to quit")
        print("=" * 60)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect (every 2nd frame for performance)
            if frame_count % 2 == 0:
                results = self.detector.detect(frame)
            
            output_frame = self.draw_results(frame, results)
            
            cv2.putText(output_frame, f"Frame: {frame_count}",
                       (output_frame.shape[1] - 150, 30),
                       self.font, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Hybrid Safety Detection', output_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped.")

if __name__ == "__main__":
    app = HybridDetectionApp()
    app.run()

