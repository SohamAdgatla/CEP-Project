"""
Data Collection Script
Helps collect and label training data from camera
"""

import cv2
import os
from pathlib import Path
import json
from datetime import datetime

class DataCollector:
    """Collect training data from camera"""
    
    def __init__(self, output_dir: str = 'raw_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)
        self.cap = None
        self.frame_count = 0
        
    def setup_camera(self):
        """Setup camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Camera initialized")
    
    def collect_images(self, save_interval: int = 30):
        """
        Collect images from camera
        
        Args:
            save_interval: Save every N frames
        """
        if self.cap is None:
            self.setup_camera()
        
        print("=" * 60)
        print("Data Collection Mode")
        print("=" * 60)
        print("Controls:")
        print("  SPACE: Save current frame")
        print("  'q': Quit")
        print(f"  Auto-save: Every {save_interval} frames")
        print("=" * 60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Display frame count
            cv2.putText(frame, f"Frame: {self.frame_count} | Press SPACE to save",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection - Press SPACE to save, Q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' ') or self.frame_count % save_interval == 0:
                # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"image_{timestamp}_{self.frame_count:06d}.jpg"
                filepath = self.images_dir / filename
                cv2.imwrite(str(filepath), frame)
                print(f"Saved: {filename}")
            
            self.frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nCollection complete! Images saved to: {self.images_dir}")
        print(f"Total images collected: {len(list(self.images_dir.glob('*.jpg')))}")
        print("\nNext steps:")
        print("1. Label images using LabelImg or similar tool")
        print("2. Use prepare_dataset.py to organize the dataset")
        print("3. Train model using train_high_accuracy_model.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training data from camera')
    parser.add_argument('--output', type=str, default='raw_data',
                       help='Output directory for collected images')
    parser.add_argument('--interval', type=int, default=30,
                       help='Auto-save interval (frames)')
    
    args = parser.parse_args()
    
    collector = DataCollector(args.output)
    collector.collect_images(save_interval=args.interval)

