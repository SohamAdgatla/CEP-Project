"""
Model Validation Script
Comprehensive validation and testing of trained model
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np
from typing import Dict, List

class ModelValidator:
    """Validate trained model performance"""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.results = {}
        
    def validate_on_dataset(self, dataset_yaml: str, split: str = 'val'):
        """Validate model on dataset"""
        print(f"Validating on {split} set...")
        
        results = self.model.val(
            data=dataset_yaml,
            split=split,
            imgsz=640,
            conf=0.5,
            iou=0.5,
            plots=True,
            save_json=True,
            verbose=True
        )
        
        self.results[split] = results
        return results
    
    def test_on_images(self, images_dir: str, conf_threshold: float = 0.5):
        """Test model on individual images"""
        images_path = Path(images_dir)
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        print(f"Testing on {len(image_files)} images...")
        
        results = []
        for img_file in image_files:
            result = self.model(str(img_file), conf=conf_threshold, verbose=False)
            results.append({
                'image': str(img_file),
                'detections': len(result[0].boxes) if result[0].boxes is not None else 0,
                'result': result[0]
            })
        
        return results
    
    def test_realtime(self, conf_threshold: float = 0.5):
        """Test model in real-time"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Real-time testing mode")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Run detection
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            cv2.imshow('Model Validation - Real-time', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def print_metrics(self):
        """Print validation metrics"""
        print("\n" + "=" * 70)
        print("MODEL VALIDATION METRICS")
        print("=" * 70)
        
        for split, result in self.results.items():
            print(f"\n{split.upper()} Set Results:")
            if hasattr(result, 'results_dict'):
                metrics = result.results_dict
                print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
                print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
                print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
                print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
        
        print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--dataset', type=str,
                       help='Path to dataset YAML for validation')
    parser.add_argument('--images', type=str,
                       help='Directory with test images')
    parser.add_argument('--realtime', action='store_true',
                       help='Test in real-time mode')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        exit(1)
    
    validator = ModelValidator(args.model)
    
    if args.dataset:
        validator.validate_on_dataset(args.dataset)
        validator.print_metrics()
    
    if args.images:
        results = validator.test_on_images(args.images, conf_threshold=args.conf)
        print(f"\nTested {len(results)} images")
        total_detections = sum(r['detections'] for r in results)
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/len(results):.2f}")
    
    if args.realtime:
        validator.test_realtime(conf_threshold=args.conf)

