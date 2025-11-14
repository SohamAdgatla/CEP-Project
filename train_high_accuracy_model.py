"""
High-Accuracy PPE Detection Model Training Script
Optimized for maximum accuracy with comprehensive data augmentation and best practices
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path
import torch

class HighAccuracyTrainer:
    """Trainer optimized for maximum accuracy"""
    
    def __init__(self, data_yaml: str, model_size: str = 'm'):
        """
        Initialize trainer
        
        Args:
            data_yaml: Path to dataset YAML file
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
                       Larger = more accurate but slower
        """
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.model = None
        
    def create_model(self):
        """Create YOLOv8 model"""
        model_name = f'yolov8{self.model_size}.pt'
        print(f"Loading model: {model_name}")
        self.model = YOLO(model_name)
        return self.model
    
    def train(self, 
              epochs: int = 300,
              imgsz: int = 640,
              batch: int = 16,
              patience: int = 100,
              save_period: int = 10,
              project_name: str = 'ppe_detector_high_accuracy'):
        """
        Train model with optimal settings for high accuracy
        
        Args:
            epochs: Maximum training epochs
            imgsz: Image size (640, 1280 for higher accuracy)
            batch: Batch size (adjust based on GPU memory)
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            project_name: Project name for saving results
        """
        if self.model is None:
            self.create_model()
        
        print("=" * 70)
        print("HIGH-ACCURACY PPE DETECTION MODEL TRAINING")
        print("=" * 70)
        print(f"Dataset: {self.data_yaml}")
        print(f"Model: YOLOv8{self.model_size}")
        print(f"Image Size: {imgsz}")
        print(f"Batch Size: {batch}")
        print(f"Max Epochs: {epochs}")
        print("=" * 70)
        
        # Training with comprehensive augmentation and optimization
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            save_period=save_period,
            project=project_name,
            name='train',
            
            # Data augmentation for robustness
            hsv_h=0.015,      # Hue augmentation
            hsv_s=0.7,        # Saturation augmentation
            hsv_v=0.4,        # Value augmentation
            degrees=10.0,     # Rotation augmentation
            translate=0.1,    # Translation augmentation
            scale=0.5,        # Scale augmentation
            shear=2.0,        # Shear augmentation
            perspective=0.0,  # Perspective augmentation
            flipud=0.0,       # Vertical flip (usually not needed for PPE)
            fliplr=0.5,       # Horizontal flip
            mosaic=1.0,       # Mosaic augmentation
            mixup=0.1,        # Mixup augmentation
            copy_paste=0.1,   # Copy-paste augmentation
            
            # Optimization settings
            optimizer='AdamW',  # AdamW optimizer for better convergence
            lr0=0.001,          # Initial learning rate
            lrf=0.01,           # Final learning rate (lr0 * lrf)
            momentum=0.937,      # SGD momentum
            weight_decay=0.0005, # Weight decay
            warmup_epochs=3.0,   # Warmup epochs
            warmup_momentum=0.8, # Warmup momentum
            warmup_bias_lr=0.1,  # Warmup bias learning rate
            
            # Loss function weights (tuned for object detection)
            box=7.5,             # Box loss gain
            cls=0.5,             # Class loss gain
            dfl=1.5,             # DFL loss gain
            
            # Training settings
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=8,           # Data loading workers
            amp=True,            # Automatic Mixed Precision
            fraction=1.0,        # Dataset fraction to use
            profile=False,       # Profile ONNX and TensorRT speeds
            freeze=None,         # Freeze layers
            multi_scale=False,   # Multi-scale training
            
            # Validation settings
            val=True,            # Validate during training
            plots=True,          # Save plots
            save=True,           # Save checkpoints
            save_json=False,     # Save results to JSON
            verbose=True,        # Verbose output
            seed=0,              # Random seed
            deterministic=True,  # Deterministic mode
            single_cls=False,    # Single class training
            rect=False,          # Rectangular training
            cos_lr=False,        # Cosine LR scheduler
            close_mosaic=10,     # Disable mosaic augmentation for last N epochs
            resume=False,        # Resume training
            pretrained=True,     # Use pretrained weights
        )
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED!")
        print("=" * 70)
        print(f"Best model saved to: {results.save_dir}/weights/best.pt")
        print(f"Last model saved to: {results.save_dir}/weights/last.pt")
        print("\nMetrics:")
        print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print("=" * 70)
        
        return results

def create_dataset_yaml(dataset_path: str, output_file: str = 'dataset.yaml'):
    """Create dataset YAML file"""
    dataset_path = Path(dataset_path)
    
    yaml_content = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 5,
        'names': {
            0: 'helmet',
            1: 'vest',
            2: 'gloves',
            3: 'safety_glasses',
            4: 'person'
        }
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset YAML created: {output_file}")
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train high-accuracy PPE detection model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, choices=[640, 1280],
                       help='Image size (1280 for higher accuracy)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--create-yaml', action='store_true',
                       help='Create dataset YAML from dataset path')
    
    args = parser.parse_args()
    
    # Create YAML if requested
    if args.create_yaml:
        data_yaml = create_dataset_yaml(args.data)
    else:
        data_yaml = args.data
    
    # Verify dataset exists
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset YAML not found: {data_yaml}")
        print("\nTo create dataset YAML, use:")
        print(f"  python train_high_accuracy_model.py --data ./dataset --create-yaml")
        exit(1)
    
    # Create trainer and train
    trainer = HighAccuracyTrainer(data_yaml, model_size=args.model)
    trainer.create_model()
    
    print("\nStarting training...")
    print("This may take several hours depending on dataset size and hardware.")
    print("For best accuracy, use:")
    print("  - Large model (--model l or x)")
    print("  - High resolution (--imgsz 1280)")
    print("  - Many epochs (--epochs 300+)")
    print("\nPress Ctrl+C to stop training early (model will be saved)\n")
    
    try:
        results = trainer.train(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch
        )
        
        print("\n" + "=" * 70)
        print("SUCCESS! Model training completed.")
        print("=" * 70)
        print(f"\nTo use the trained model:")
        print(f"  python improved_detector.py {results.save_dir}/weights/best.pt")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Check for saved checkpoints in the runs/ directory.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

