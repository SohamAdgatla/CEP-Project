"""
Dataset Finder Script
Helps find and download public PPE detection datasets
"""

import os
import requests
from pathlib import Path
import json

class DatasetFinder:
    """Find and help download PPE datasets"""
    
    def __init__(self):
        self.datasets_info = {
            'roboflow_ppe': {
                'name': 'Roboflow PPE Detection',
                'url': 'https://universe.roboflow.com/roboflow-universe/ppe-personal-protective-equipment-detection',
                'description': 'Large dataset with PPE annotations',
                'format': 'YOLO',
                'classes': ['helmet', 'vest', 'gloves', 'safety_glasses', 'person'],
                'size': '1000+ images',
                'instructions': [
                    '1. Visit: https://universe.roboflow.com/roboflow-universe/ppe-personal-protective-equipment-detection',
                    '2. Sign up for free account',
                    '3. Click "Download"',
                    '4. Select "YOLO" format',
                    '5. Download and extract',
                    '6. Use prepare_dataset.py to organize'
                ]
            },
            'kaggle_ppe': {
                'name': 'Kaggle PPE Datasets',
                'url': 'https://www.kaggle.com/datasets?search=ppe',
                'description': 'Various PPE detection datasets',
                'format': 'Various',
                'classes': 'Varies',
                'size': 'Varies',
                'instructions': [
                    '1. Visit: https://www.kaggle.com/datasets?search=ppe',
                    '2. Browse available datasets',
                    '3. Download dataset',
                    '4. Convert to YOLO format if needed',
                    '5. Use prepare_dataset.py to organize'
                ]
            },
            'google_open_images': {
                'name': 'Google Open Images (with PPE)',
                'url': 'https://storage.googleapis.com/openimages/web/index.html',
                'description': 'Large open dataset, search for PPE classes',
                'format': 'Various',
                'classes': 'Search for: hard hat, safety vest, etc.',
                'size': 'Very large',
                'instructions': [
                    '1. Visit: https://storage.googleapis.com/openimages/web/index.html',
                    '2. Search for PPE-related classes',
                    '3. Download images and annotations',
                    '4. Convert to YOLO format',
                    '5. Use prepare_dataset.py to organize'
                ]
            }
        }
    
    def list_datasets(self):
        """List available datasets"""
        print("=" * 70)
        print("AVAILABLE PPE DETECTION DATASETS")
        print("=" * 70)
        
        for key, info in self.datasets_info.items():
            print(f"\n{info['name']}")
            print(f"  URL: {info['url']}")
            print(f"  Description: {info['description']}")
            print(f"  Format: {info['format']}")
            print(f"  Classes: {info['classes']}")
            print(f"  Size: {info['size']}")
            print(f"  Instructions:")
            for instruction in info['instructions']:
                print(f"    {instruction}")
        
        print("\n" + "=" * 70)
        print("RECOMMENDED: Roboflow PPE Detection Dataset")
        print("=" * 70)
        print("This is the easiest to use and comes in YOLO format.")
        print("Visit: https://universe.roboflow.com/roboflow-universe/ppe-personal-protective-equipment-detection")
    
    def create_download_script(self, dataset_type: str = 'roboflow'):
        """Create a script to help download dataset"""
        if dataset_type not in self.datasets_info:
            print(f"Unknown dataset type: {dataset_type}")
            return
        
        info = self.datasets_info[dataset_type]
        
        script_content = f"""#!/bin/bash
# Dataset Download Helper Script
# Dataset: {info['name']}

echo "Dataset: {info['name']}"
echo "URL: {info['url']}"
echo ""
echo "Please follow these steps:"
"""
        
        for i, instruction in enumerate(info['instructions'], 1):
            script_content += f'echo "{instruction}"\n'
        
        script_content += """
echo ""
echo "After downloading, run:"
echo "  python prepare_dataset.py --source-images <images_dir> --source-labels <labels_dir> --output dataset"
"""
        
        script_path = Path('download_dataset.sh')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Download helper script created: {script_path}")
        print(f"\nTo download {info['name']}:")
        print(f"  Visit: {info['url']}")
        for instruction in info['instructions']:
            print(f"  {instruction}")

def main():
    finder = DatasetFinder()
    finder.list_datasets()
    
    print("\n" + "=" * 70)
    print("QUICK START")
    print("=" * 70)
    print("\nOption 1: Use Roboflow Dataset (Recommended)")
    print("  1. Visit: https://universe.roboflow.com/roboflow-universe/ppe-personal-protective-equipment-detection")
    print("  2. Sign up and download in YOLO format")
    print("  3. Extract to a folder")
    print("  4. Run: python prepare_dataset.py --source-images <path>/train/images --source-labels <path>/train/labels --output dataset")
    print("\nOption 2: Collect Your Own Data")
    print("  1. Run: python collect_training_data.py")
    print("  2. Label images using LabelImg")
    print("  3. Run: python prepare_dataset.py")
    print("\nOption 3: Use Multiple Sources")
    print("  Combine datasets from multiple sources for maximum diversity")

if __name__ == "__main__":
    main()

