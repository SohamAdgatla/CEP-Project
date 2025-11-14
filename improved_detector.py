"""
Improved Safety Kit Detection System
Enhanced with better false positive prevention and detection accuracy
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import time
import os
from collections import deque

class AdvancedImageEnhancer:
    """Advanced image enhancement for better detection clarity"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply comprehensive image enhancement"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance luminance channel
        l = self.clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Denoise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Unsharp masking for sharpness
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Gamma correction for better visibility
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced

class FalsePositiveFilter:
    """Advanced false positive filtering system"""
    
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.detection_history = {
            'helmet': deque(maxlen=history_size),
            'vest': deque(maxlen=history_size),
            'gloves': deque(maxlen=history_size),
            'safety_glasses': deque(maxlen=history_size)
        }
        
    def analyze_helmet(self, roi: np.ndarray, bbox: Tuple) -> bool:
        """Analyze if detection is actually a helmet (not hair)"""
        if roi.size == 0:
            return False
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Helmet characteristics:
        # 1. More uniform color (less variation than hair)
        std_dev = np.std(gray)
        if std_dev > 35:  # Hair has more variation
            return False
        
        # 2. Hard edges (helmet has defined edges, hair is soft)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Hair typically has higher edge density
        if edge_density > 0.25:
            return False
        
        # 3. Shape analysis - helmet is typically more circular/oval
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Helmet should be more circular than hair
                if circularity < 0.3:
                    return False
        
        # 4. Color consistency - helmet usually has uniform color
        mean_color = np.mean(gray)
        if mean_color < 30 or mean_color > 200:  # Too dark or too bright
            return False
        
        return True
    
    def analyze_vest(self, roi: np.ndarray, bbox: Tuple) -> bool:
        """Analyze if detection is actually a safety vest (not regular shirt)"""
        if roi.size == 0:
            return False
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        
        # Safety vest characteristics:
        # 1. Bright yellow or orange color
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_ratio = np.sum(yellow_mask > 0) / (h * w)
        
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([20, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        orange_ratio = np.sum(orange_mask > 0) / (h * w)
        
        # Check for bright/reflective materials
        v_channel = hsv[:, :, 2]  # Value channel
        brightness = np.mean(v_channel)
        
        # Safety vest should have significant yellow/orange OR high brightness
        if yellow_ratio < 0.25 and orange_ratio < 0.25:
            if brightness < 120:  # Not bright enough
                return False
        
        # 2. Check for reflective strips (high brightness areas)
        bright_areas = cv2.threshold(v_channel, 200, 255, cv2.THRESH_BINARY)[1]
        bright_ratio = np.sum(bright_areas > 0) / (h * w)
        
        # Safety vests often have reflective strips
        if bright_ratio > 0.1 or yellow_ratio > 0.3 or orange_ratio > 0.3:
            return True
        
        return False
    
    def analyze_safety_glasses(self, roi: np.ndarray, bbox: Tuple) -> bool:
        """Analyze if detection is actually safety glasses (not beard/face)"""
        if roi.size == 0:
            return False
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Safety glasses characteristics:
        # 1. Distinct frame structure with horizontal elements
        edges = cv2.Canny(gray, 50, 150)
        
        # Check for horizontal lines (glasses frame)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//2, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_ratio = np.sum(horizontal_lines > 0) / (h * w)
        
        if horizontal_ratio < 0.05:  # No clear horizontal frame structure
            return False
        
        # 2. Check for lens areas (darker regions with defined edges)
        # Glasses have distinct lens areas
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:  # Glasses typically have 2 lens areas
            return False
        
        # 3. Edge density should be moderate (frame structure, not dense like beard)
        edge_density = np.sum(edges > 0) / (h * w)
        if edge_density < 0.08 or edge_density > 0.35:
            return False
        
        # 4. Check for symmetry (glasses are typically symmetric)
        left_half = edges[:, :w//2]
        right_half = edges[:, w//2:]
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize if needed for comparison
        if left_half.shape != right_half_flipped.shape:
            right_half_flipped = cv2.resize(right_half_flipped, 
                                          (left_half.shape[1], left_half.shape[0]))
        
        # Calculate similarity
        if left_half.shape == right_half_flipped.shape:
            similarity = cv2.matchTemplate(left_half, right_half_flipped, cv2.TM_CCOEFF_NORMED)[0][0]
            if similarity < 0.3:  # Not symmetric enough
                return False
        
        return True
    
    def analyze_gloves(self, roi: np.ndarray, bbox: Tuple) -> bool:
        """Analyze if detection is actually gloves"""
        if roi.size == 0:
            return False
        
        # Gloves are harder to distinguish, rely more on:
        # 1. Position (should be in hand region)
        # 2. Texture analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Gloves typically have:
        # - Moderate texture (not smooth like skin, not rough like fabric)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / gray.size
        
        # Gloves should have some texture
        if edge_density < 0.05 or edge_density > 0.4:
            return False
        
        return True
    
    def validate_detection(self, class_name: str, roi: np.ndarray, bbox: Tuple) -> bool:
        """Validate detection based on class-specific analysis"""
        class_name_lower = class_name.lower()
        
        if 'helmet' in class_name_lower or 'hard hat' in class_name_lower:
            return self.analyze_helmet(roi, bbox)
        elif 'vest' in class_name_lower or 'safety vest' in class_name_lower:
            return self.analyze_vest(roi, bbox)
        elif 'glass' in class_name_lower or 'goggle' in class_name_lower:
            return self.analyze_safety_glasses(roi, bbox)
        elif 'glove' in class_name_lower:
            return self.analyze_gloves(roi, bbox)
        
        return True  # Default: accept if we can't validate
    
    def check_temporal_consistency(self, class_name: str, confidence: float) -> bool:
        """Check if detection is consistent over time"""
        class_name_lower = class_name.lower()
        key = None
        
        for k in self.detection_history.keys():
            if k in class_name_lower:
                key = k
                break
        
        if key is None:
            return True
        
        # Add to history
        self.detection_history[key].append({
            'confidence': confidence,
            'time': time.time()
        })
        
        # Check consistency
        if len(self.detection_history[key]) < 3:
            # Need at least 3 detections for consistency check
            return confidence > 0.75  # Higher threshold for single detections
        
        # Check if consistently detected
        recent = list(self.detection_history[key])[-5:]
        confidences = [d['confidence'] for d in recent]
        avg_confidence = np.mean(confidences)
        
        # Must have consistent high confidence
        return avg_confidence >= 0.65 and confidence >= 0.6

class ImprovedSafetyDetector:
    """Improved safety detector with advanced false positive prevention"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.enhancer = AdvancedImageEnhancer()
        self.false_positive_filter = FalsePositiveFilter()
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom model from: {model_path}")
        else:
            # Use general YOLO model (will need custom training for PPE)
            self.model = YOLO('yolov8n.pt')
            print("WARNING: Using general YOLO model.")
            print("For accurate PPE detection, train a custom model using train_ppe_model.py")
            print("See README.md for instructions.")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """Detect safety equipment with advanced filtering"""
        # Enhance image
        enhanced = self.enhancer.enhance(frame)
        
        # Run detection with higher confidence threshold
        results = self.model(enhanced, conf=self.confidence_threshold, verbose=False)
        
        detected_equipment = {
            'helmet': False,
            'vest': False,
            'gloves': False,
            'safety_glasses': False,
            'person': False
        }
        
        valid_detections = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                class_name = result.names[cls]
                
                # Extract ROI for validation
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                roi = enhanced[y1:y2, x1:x2]
                
                # Validate detection
                if self.false_positive_filter.validate_detection(class_name, roi, bbox):
                    # Check temporal consistency
                    if self.false_positive_filter.check_temporal_consistency(class_name, conf):
                        valid_detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': bbox
                        })
                        
                        # Update equipment status
                        class_lower = class_name.lower()
                        if 'helmet' in class_lower or 'hard hat' in class_lower:
                            detected_equipment['helmet'] = True
                        elif 'vest' in class_lower:
                            detected_equipment['vest'] = True
                        elif 'glove' in class_lower:
                            detected_equipment['gloves'] = True
                        elif 'glass' in class_lower or 'goggle' in class_lower:
                            detected_equipment['safety_glasses'] = True
                        elif 'person' in class_lower:
                            detected_equipment['person'] = True
        
        return {
            'equipment': detected_equipment,
            'detections': valid_detections
        }

class SafetyDetectionApp:
    """Main application with improved detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.detector = ImprovedSafetyDetector(model_path=model_path, confidence_threshold=0.7)
        self.cap = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_detections(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results with improved visualization"""
        frame_copy = frame.copy()
        
        # Draw bounding boxes
        for det in results['detections']:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det['class']
            
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Label with confidence
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, self.font, 0.6, 2)
            cv2.rectangle(frame_copy, (x1, y1 - label_h - 10),
                         (x1 + label_w, y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                       self.font, 0.6, (0, 0, 0), 2)
        
        # Status panel
        equipment = results['equipment']
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 320, 180
        
        # Semi-transparent background
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
        
        # Border
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
        overall_status = "SAFE" if all_detected else "UNSAFE - PPE REQUIRED"
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
        
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
        
        print("=" * 60)
        print("Safety Kit Detection System - Improved Version")
        print("=" * 60)
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("=" * 60)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect (every frame for real-time, or every N frames for performance)
            if frame_count % 2 == 0:  # Process every 2nd frame for better performance
                results = self.detector.detect(frame)
            
            # Draw results
            output_frame = self.draw_detections(frame, results)
            
            # Add FPS counter
            cv2.putText(output_frame, f"Frame: {frame_count}",
                       (output_frame.shape[1] - 150, 30),
                       self.font, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Safety Kit Detection System', output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"safety_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, output_frame)
                print(f"Frame saved as {filename}")
            
            frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped.")

if __name__ == "__main__":
    import sys
    
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    app = SafetyDetectionApp(model_path=model_path)
    app.run()

