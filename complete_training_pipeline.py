"""
Complete Training Pipeline
Automated end-to-end training process for maximum accuracy
"""

import os
import sys
from pathlib import Path
import subprocess

class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self, project_name: str = "ppe_detector"):
        self.project_name = project_name
        self.steps_completed = []
        
    def check_requirements(self):
        """Check if all requirements are met"""
        print("=" * 70)
        print("CHECKING REQUIREMENTS")
        print("=" * 70)
        
        checks = {
            'Python': self._check_python(),
            'CUDA/GPU': self._check_gpu(),
            'Dependencies': self._check_dependencies(),
        }
        
        all_ok = all(checks.values())
        
        print("\nRequirements Status:")
        for check, status in checks.items():
            status_str = "✓" if status else "✗"
            print(f"  {status_str} {check}")
        
        if not all_ok:
            print("\n⚠ Some requirements not met. Training may be slower.")
        
        return all_ok
    
    def _check_python(self):
        """Check Python version"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
            return True
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False
    
    def _check_gpu(self):
        """Check GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✓ GPU detected: {gpu_name}")
                return True
            else:
                print("✗ No GPU detected (CPU training will be slow)")
                return False
        except:
            print("✗ Could not check GPU")
            return False
    
    def _check_dependencies(self):
        """Check if dependencies are installed"""
        try:
            import cv2
            import ultralytics
            import numpy
            print("✓ All dependencies installed")
            return True
        except ImportError as e:
            print(f"✗ Missing dependency: {e}")
            return False
    
    def run_pipeline(self, 
                    dataset_yaml: str,
                    model_size: str = 'l',
                    epochs: int = 500,
                    imgsz: int = 1280,
                    batch: int = 8):
        """
        Run complete training pipeline
        
        Args:
            dataset_yaml: Path to dataset YAML
            model_size: Model size (n, s, m, l, x)
            epochs: Number of epochs
            imgsz: Image size
            batch: Batch size
        """
        print("\n" + "=" * 70)
        print("COMPLETE TRAINING PIPELINE")
        print("=" * 70)
        print(f"Dataset: {dataset_yaml}")
        print(f"Model: YOLOv8{model_size}")
        print(f"Epochs: {epochs}")
        print(f"Image Size: {imgsz}")
        print(f"Batch Size: {batch}")
        print("=" * 70)
        
        # Check requirements
        if not self.check_requirements():
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Pipeline cancelled.")
                return
        
        # Verify dataset
        if not os.path.exists(dataset_yaml):
            print(f"\n✗ Error: Dataset YAML not found: {dataset_yaml}")
            print("\nTo create dataset:")
            print("1. Collect images: python collect_training_data.py")
            print("2. Label images: Use LabelImg or Roboflow")
            print("3. Prepare dataset: python prepare_dataset.py")
            return
        
        print(f"\n✓ Dataset found: {dataset_yaml}")
        
        # Start training
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print("This will take several hours. Progress will be shown below.")
        print("Press Ctrl+C to stop (model will be saved)")
        print("=" * 70 + "\n")
        
        try:
            # Run training
            cmd = [
                sys.executable,
                'train_high_accuracy_model.py',
                '--data', dataset_yaml,
                '--model', model_size,
                '--epochs', str(epochs),
                '--imgsz', str(imgsz),
                '--batch', str(batch)
            ]
            
            result = subprocess.run(cmd, check=True)
            
            # Find best model
            model_dir = Path('runs') / 'detect' / 'ppe_detector_high_accuracy' / 'train' / 'weights'
            best_model = model_dir / 'best.pt'
            
            if best_model.exists():
                print("\n" + "=" * 70)
                print("TRAINING COMPLETED SUCCESSFULLY!")
                print("=" * 70)
                print(f"Best model: {best_model}")
                print("\nNext steps:")
                print(f"1. Validate: python validate_model.py --model {best_model} --dataset {dataset_yaml}")
                print(f"2. Test real-time: python validate_model.py --model {best_model} --realtime")
                print(f"3. Use model: python improved_detector.py {best_model}")
                print("=" * 70)
            else:
                print("\n⚠ Training completed but model not found at expected location")
                print("Check runs/detect/ directory for results")
        
        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted by user")
            print("Check runs/detect/ directory for partial results")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Training failed with error: {e}")
            print("Check error messages above for details")
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Complete training pipeline for maximum accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Maximum accuracy training
  python complete_training_pipeline.py --data dataset/dataset.yaml --model l --epochs 500 --imgsz 1280
  
  # Faster training (good accuracy)
  python complete_training_pipeline.py --data dataset/dataset.yaml --model m --epochs 300 --imgsz 640
        """
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='l',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (l=large recommended for accuracy)')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs (500 recommended)')
    parser.add_argument('--imgsz', type=int, default=1280,
                       choices=[640, 1280],
                       help='Image size (1280 for maximum accuracy)')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size (adjust for GPU memory)')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    pipeline.run_pipeline(
        dataset_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )

if __name__ == "__main__":
    main()

