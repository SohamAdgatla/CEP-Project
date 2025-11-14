"""
Dataset Preparation Script
Helps organize and prepare dataset for training
"""

import os
import shutil
from pathlib import Path
import random
from typing import List, Tuple
import cv2
import yaml

class DatasetPreparer:
    """Prepare and organize dataset for YOLO training"""
    
    def __init__(self, source_dir: str, output_dir: str = 'dataset'):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1
        
    def create_structure(self):
        """Create dataset directory structure"""
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'images' / 'test',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val',
            self.output_dir / 'labels' / 'test',
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Created dataset structure in: {self.output_dir}")
    
    def split_dataset(self, images_dir: str, labels_dir: str):
        """
        Split dataset into train/val/test
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing labels (YOLO format)
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_path.glob(f'*{ext}')))
            image_files.extend(list(images_path.glob(f'*{ext.upper()}')))
        
        # Shuffle
        random.seed(42)
        random.shuffle(image_files)
        
        total = len(image_files)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        print(f"\nDataset split:")
        print(f"  Total images: {total}")
        print(f"  Train: {len(train_files)} ({len(train_files)/total*100:.1f}%)")
        print(f"  Val: {len(val_files)} ({len(val_files)/total*100:.1f}%)")
        print(f"  Test: {len(test_files)} ({len(test_files)/total*100:.1f}%)")
        
        # Copy files
        splits = [
            ('train', train_files),
            ('val', val_files),
            ('test', test_files)
        ]
        
        for split_name, files in splits:
            print(f"\nCopying {split_name} files...")
            for img_file in files:
                # Copy image
                dst_img = self.output_dir / 'images' / split_name / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy corresponding label
                label_file = labels_path / (img_file.stem + '.txt')
                if label_file.exists():
                    dst_label = self.output_dir / 'labels' / split_name / label_file.name
                    shutil.copy2(label_file, dst_label)
                else:
                    print(f"  Warning: Label not found for {img_file.name}")
        
        print("\nDataset preparation complete!")
    
    def create_yaml(self):
        """Create dataset YAML file"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
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
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"\nDataset YAML created: {yaml_path}")
        return yaml_path
    
    def validate_dataset(self):
        """Validate dataset structure and files"""
        print("\nValidating dataset...")
        
        issues = []
        
        # Check structure
        required_dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.output_dir / dir_name
            if not dir_path.exists():
                issues.append(f"Missing directory: {dir_name}")
        
        # Check image-label correspondence
        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / 'images' / split
            label_dir = self.output_dir / 'labels' / split
            
            if img_dir.exists() and label_dir.exists():
                images = set(f.stem for f in img_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png'])
                labels = set(f.stem for f in label_dir.glob('*.txt'))
                
                missing_labels = images - labels
                missing_images = labels - images
                
                if missing_labels:
                    issues.append(f"{split}: {len(missing_labels)} images without labels")
                if missing_images:
                    issues.append(f"{split}: {len(missing_images)} labels without images")
        
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ Dataset structure is valid!")
        
        return len(issues) == 0
    
    def get_statistics(self):
        """Get dataset statistics"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            img_dir = self.output_dir / 'images' / split
            label_dir = self.output_dir / 'labels' / split
            
            if img_dir.exists():
                image_count = len(list(img_dir.glob('*')))
                label_count = len(list(label_dir.glob('*.txt'))) if label_dir.exists() else 0
                
                # Count objects per class
                class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                if label_dir.exists():
                    for label_file in label_dir.glob('*.txt'):
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_id = int(parts[0])
                                    if class_id in class_counts:
                                        class_counts[class_id] += 1
                
                stats[split] = {
                    'images': image_count,
                    'labels': label_count,
                    'classes': class_counts
                }
        
        return stats

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--source-images', type=str, required=True,
                       help='Source directory containing images')
    parser.add_argument('--source-labels', type=str, required=True,
                       help='Source directory containing labels (YOLO format)')
    parser.add_argument('--output', type=str, default='dataset',
                       help='Output directory for organized dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing dataset')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(args.source_images, args.output)
    preparer.train_ratio = args.train_ratio
    preparer.val_ratio = args.val_ratio
    preparer.test_ratio = 1.0 - args.train_ratio - args.val_ratio
    
    if args.validate_only:
        preparer.validate_dataset()
        stats = preparer.get_statistics()
        print("\nDataset Statistics:")
        for split, data in stats.items():
            print(f"\n{split.upper()}:")
            print(f"  Images: {data['images']}")
            print(f"  Labels: {data['labels']}")
            print(f"  Class distribution:")
            class_names = ['helmet', 'vest', 'gloves', 'safety_glasses', 'person']
            for class_id, count in data['classes'].items():
                print(f"    {class_names[class_id]}: {count}")
    else:
        preparer.create_structure()
        preparer.split_dataset(args.source_images, args.source_labels)
        preparer.create_yaml()
        preparer.validate_dataset()
        
        stats = preparer.get_statistics()
        print("\nDataset Statistics:")
        for split, data in stats.items():
            print(f"\n{split.upper()}:")
            print(f"  Images: {data['images']}")
            print(f"  Labels: {data['labels']}")

if __name__ == "__main__":
    main()

