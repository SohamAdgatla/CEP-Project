"""
Safety Kit Detection System
Detects PPE (Personal Protective Equipment) on workers in real-time
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import time

class ImageEnhancer:
    """Enhances camera feed for better clarity and detection accuracy"""
    
    @staticmethod
    def enhance_image(frame: np.ndarray) -> np.ndarray:
        """Apply multiple enhancement techniques to improve image quality"""
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Sharpen the image
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

class SafetyDetector:
    """Main class for detecting safety equipment"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.6):
        """
        Initialize the safety detector
        
        Args:
            model_path: Path to custom YOLO model (if None, uses pre-trained)
            confidence_threshold: Minimum confidence for detections (higher = more strict)
        """
        self.confidence_threshold = confidence_threshold
        self.enhancer = ImageEnhancer()
        
        # Initialize YOLO model
        # Using YOLOv8 which has good performance
        # For production, you would use a custom-trained model
        if model_path:
            self.model = YOLO(model_path)
        else:
            # Using YOLOv8n (nano) for speed, can be upgraded to YOLOv8s/m/l/x
            self.model = YOLO('yolov8n.pt')
            print("Note: Using general YOLO model. For best results, train a custom PPE detection model.")
        
        # Safety equipment classes (will be mapped from model predictions)
        self.safety_classes = {
            'helmet': 0,
            'vest': 1,
            'gloves': 2,
            'safety_glasses': 3,
            'person': 4
        }
        
        # Track detection history to filter false positives
        self.detection_history = {key: [] for key in self.safety_classes.keys()}
        self.history_size = 5
        
    def is_valid_detection(self, class_name: str, bbox: Tuple, frame: np.ndarray) -> bool:
        """
        Validate detection to prevent false positives
        
        Args:
            class_name: Name of detected class
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            frame: Current frame
            
        Returns:
            True if detection is valid, False otherwise
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        # Extract region of interest
        h, w = roi.shape[:2]
        if h < 10 or w < 10:
            return False
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Specific validation for each equipment type
        if class_name == 'helmet':
            # Helmet should have specific characteristics:
            # - More uniform color distribution (not like hair)
            # - Hard edges (not soft like hair)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            # Helmet typically has lower edge density than hair
            # and more uniform color
            std_dev = np.std(gray)
            
            # If too much variation (like hair) or too many edges, likely false positive
            if std_dev > 40 or edge_density > 0.3:
                return False
        
        elif class_name == 'vest':
            # Vest should be bright/reflective and have distinct color
            # Check for high brightness or specific color (yellow/orange)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Check for yellow/orange colors typical of safety vests
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            yellow_ratio = np.sum(yellow_mask > 0) / (h * w)
            
            # Check for orange colors
            orange_lower = np.array([10, 100, 100])
            orange_upper = np.array([20, 255, 255])
            orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
            orange_ratio = np.sum(orange_mask > 0) / (h * w)
            
            # If not enough yellow/orange, likely a regular shirt
            if yellow_ratio < 0.3 and orange_ratio < 0.3:
                # Check brightness - safety vests are often bright
                brightness = np.mean(gray)
                if brightness < 100:  # Too dark for a safety vest
                    return False
        
        elif class_name == 'safety_glasses':
            # Safety glasses should have:
            # - Transparent/reflective areas
            # - Distinct frame structure
            # - Not just a beard or face
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (h * w)
            
            # Glasses should have moderate edge density (frame structure)
            # Too low = might be face, too high = might be beard/hair
            if edge_density < 0.1 or edge_density > 0.4:
                return False
            
            # Check for horizontal lines (typical of glasses frame)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//3, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            if np.sum(horizontal_lines > 0) < 10:
                return False
        
        elif class_name == 'gloves':
            # Gloves are harder to distinguish, but should be:
            # - In hand region
            # - Have distinct texture
            # For now, rely on confidence threshold and position
            pass
        
        return True
    
    def filter_false_positives(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter out false positive detections using temporal consistency
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for det in detections:
            class_name = det['class']
            confidence = det['confidence']
            
            # Add to history
            self.detection_history[class_name].append({
                'confidence': confidence,
                'bbox': det['bbox'],
                'time': time.time()
            })
            
            # Keep only recent history
            current_time = time.time()
            self.detection_history[class_name] = [
                h for h in self.detection_history[class_name]
                if current_time - h['time'] < 1.0  # Last 1 second
            ]
            
            # Check temporal consistency
            if len(self.detection_history[class_name]) >= 3:
                # If detected consistently in recent frames, likely valid
                recent_confidences = [h['confidence'] for h in self.detection_history[class_name][-3:]]
                avg_confidence = np.mean(recent_confidences)
                
                if avg_confidence >= self.confidence_threshold:
                    filtered.append(det)
            elif confidence >= self.confidence_threshold + 0.1:  # Higher threshold for single detections
                filtered.append(det)
        
        return filtered
    
    def detect(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Detect safety equipment in the frame
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Dictionary with detected equipment and their status
        """
        # Enhance image for better detection
        enhanced_frame = self.enhancer.enhance_image(frame)
        
        # Run YOLO detection
        results = self.model(enhanced_frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        detected_equipment = {
            'helmet': False,
            'vest': False,
            'gloves': False,
            'safety_glasses': False,
            'person': False
        }
        
        # Parse results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                # Map YOLO classes to safety equipment
                # Note: This is a placeholder - you need to train a custom model
                # or use class names from your trained model
                class_name = result.names[cls]
                
                # For now, we'll use a general approach
                # In production, use a custom-trained PPE detection model
                detection = {
                    'class': class_name,
                    'confidence': conf,
                    'bbox': bbox
                }
                
                # Validate detection
                if self.is_valid_detection(class_name, bbox, enhanced_frame):
                    detections.append(detection)
        
        # Filter false positives using temporal consistency
        filtered_detections = self.filter_false_positives(detections)
        
        # Update detected equipment status
        for det in filtered_detections:
            class_name = det['class'].lower()
            if 'helmet' in class_name or 'hard hat' in class_name:
                detected_equipment['helmet'] = True
            elif 'vest' in class_name or 'safety vest' in class_name:
                detected_equipment['vest'] = True
            elif 'glove' in class_name:
                detected_equipment['gloves'] = True
            elif 'glass' in class_name or 'goggle' in class_name:
                detected_equipment['safety_glasses'] = True
            elif 'person' in class_name:
                detected_equipment['person'] = True
        
        return {
            'equipment': detected_equipment,
            'detections': filtered_detections
        }

class SafetyDetectionApp:
    """Main application class for safety detection system"""
    
    def __init__(self):
        self.detector = SafetyDetector(confidence_threshold=0.65)
        self.cap = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_detections(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on frame"""
        frame_copy = frame.copy()
        
        # Draw bounding boxes and labels
        for det in results['detections']:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            class_name = det['class']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for detected equipment
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, self.font, 0.6, 2)
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                       self.font, 0.6, (0, 0, 0), 2)
        
        # Draw safety status panel
        equipment = results['equipment']
        panel_y = 30
        panel_x = 10
        
        # Background for status panel
        panel_height = 150
        cv2.rectangle(frame_copy, (panel_x, panel_y),
                     (panel_x + 300, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame_copy, (panel_x, panel_y),
                     (panel_x + 300, panel_y + panel_height), (255, 255, 255), 2)
        
        # Status text
        status_text = "SAFETY STATUS:"
        cv2.putText(frame_copy, status_text, (panel_x + 10, panel_y + 25),
                   self.font, 0.7, (255, 255, 255), 2)
        
        y_offset = 50
        items = [
            ('Helmet', equipment['helmet']),
            ('Safety Vest', equipment['vest']),
            ('Gloves', equipment['gloves']),
            ('Safety Glasses', equipment['safety_glasses'])
        ]
        
        for item_name, detected in items:
            color = (0, 255, 0) if detected else (0, 0, 255)
            status = "✓ DETECTED" if detected else "✗ MISSING"
            cv2.putText(frame_copy, f"{item_name}: {status}",
                       (panel_x + 10, panel_y + y_offset),
                       self.font, 0.5, color, 1)
            y_offset += 25
        
        # Overall safety status
        all_detected = all([equipment['helmet'], equipment['vest'],
                           equipment['gloves'], equipment['safety_glasses']])
        overall_color = (0, 255, 0) if all_detected else (0, 0, 255)
        overall_status = "SAFE" if all_detected else "UNSAFE"
        cv2.putText(frame_copy, f"OVERALL: {overall_status}",
                   (panel_x + 10, panel_y + y_offset + 10),
                   self.font, 0.6, overall_color, 2)
        
        return frame_copy
    
    def run(self):
        """Run the safety detection application"""
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Safety Detection System Started")
        print("Press 'q' to quit")
        print("Note: For accurate detection, train a custom PPE detection model")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect safety equipment
            results = self.detector.detect(frame)
            
            # Draw results
            output_frame = self.draw_detections(frame, results)
            
            # Display frame
            cv2.imshow('Safety Kit Detection System', output_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Safety Detection System Stopped")

if __name__ == "__main__":
    app = SafetyDetectionApp()
    app.run()

