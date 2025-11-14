"""
Training script for custom PPE (Personal Protective Equipment) detection model
This script helps you train a YOLOv8 model specifically for safety equipment detection
"""

from ultralytics import YOLO
import os

def train_ppe_model(data_path: str, epochs: int = 100, imgsz: int = 640):
    """
    Train a custom YOLOv8 model for PPE detection
    
    Args:
        data_path: Path to dataset YAML file (YOLO format)
        epochs: Number of training epochs
        imgsz: Image size for training
    """
    # Initialize model (using YOLOv8n for speed, can use YOLOv8s/m/l/x for better accuracy)
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name='ppe_detector',
        patience=50,  # Early stopping patience
        save=True,
        plots=True
    )
    
    print(f"Training completed! Model saved to: {results.save_dir}")
    return results

if __name__ == "__main__":
    print("PPE Model Training Script")
    print("=" * 50)
    print("\nTo train a custom model, you need:")
    print("1. A dataset in YOLO format with labeled images")
    print("2. A dataset.yaml file specifying:")
    print("   - Path to images")
    print("   - Number of classes")
    print("   - Class names: helmet, vest, gloves, safety_glasses, person")
    print("\nExample dataset.yaml:")
    print("""
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: helmet
  1: vest
  2: gloves
  3: safety_glasses
  4: person
    """)
    print("\nRun training with:")
    print("python train_ppe_model.py")
    print("\nOr modify this script to specify your dataset path.")

