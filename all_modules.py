"""
All Modules - Central Import File
==================================
This file contains all imports used across the Safety Kit Detection System project.
Import this file to get access to all required modules at once.

Usage:
    from all_modules import *
    # OR
    import all_modules as modules
    # Then use: modules.cv2, modules.np, etc.
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

# System and OS
import os
import sys
import subprocess
import platform

# Determine if running as script or being imported
_IS_MAIN = __name__ == "__main__"
_SHOW_WARNINGS = not _IS_MAIN  # Only show warnings when imported, not when run directly

# File and Path Operations
from pathlib import Path
import shutil

# Data Types and Collections
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque, defaultdict, Counter

# Date and Time
import time
from datetime import datetime, timedelta

# Random and Math
import random
import math

# JSON and Data Serialization
import json
import pickle

# YAML (for dataset configuration)
try:
    import yaml
except ImportError:
    yaml = None
    if _SHOW_WARNINGS:
        print("Warning: PyYAML not installed. Install with: pip install PyYAML")

# ============================================================================
# THIRD-PARTY IMPORTS - Computer Vision and Image Processing
# ============================================================================

# OpenCV - Computer Vision Library
try:
    import cv2
except ImportError:
    cv2 = None
    if _SHOW_WARNINGS:
        print("Warning: OpenCV not installed. Install with: pip install opencv-python")

# NumPy - Numerical Computing
try:
    import numpy as np
except ImportError:
    np = None
    if _SHOW_WARNINGS:
        print("Warning: NumPy not installed. Install with: pip install numpy")

# Pillow - Image Processing
try:
    from PIL import Image, ImageEnhance, ImageFilter
    import PIL
except ImportError:
    Image = None
    ImageEnhance = None
    ImageFilter = None
    PIL = None
    if _SHOW_WARNINGS:
        print("Warning: Pillow not installed. Install with: pip install Pillow")

# Scikit-Image - Image Processing
try:
    from skimage import filters, exposure, morphology, measure
    import skimage
except ImportError:
    filters = None
    exposure = None
    morphology = None
    measure = None
    skimage = None
    if _SHOW_WARNINGS:
        print("Warning: scikit-image not installed. Install with: pip install scikit-image")

# Imutils - Image Utilities
try:
    import imutils
except ImportError:
    imutils = None
    if _SHOW_WARNINGS:
        print("Warning: imutils not installed. Install with: pip install imutils")

# ============================================================================
# THIRD-PARTY IMPORTS - Machine Learning and Deep Learning
# ============================================================================

# PyTorch - Deep Learning Framework
try:
    import torch
    import torchvision
    from torch import nn, optim
    from torchvision import transforms, models, datasets
except ImportError:
    torch = None
    torchvision = None
    nn = None
    optim = None
    transforms = None
    models = None
    datasets = None
    if _SHOW_WARNINGS:
        print("Warning: PyTorch not installed. Install with: pip install torch torchvision")

# Ultralytics YOLO - Object Detection
try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, colorstr
    from ultralytics.models import YOLO as YOLOModel
except ImportError:
    YOLO = None
    LOGGER = None
    colorstr = None
    YOLOModel = None
    if _SHOW_WARNINGS:
        print("Warning: Ultralytics not installed. Install with: pip install ultralytics")

# ============================================================================
# THIRD-PARTY IMPORTS - Utilities
# ============================================================================

# Requests - HTTP Library (for dataset downloading)
try:
    import requests
    from requests.exceptions import RequestException, Timeout
except ImportError:
    requests = None
    RequestException = None
    Timeout = None
    if _SHOW_WARNINGS:
        print("Warning: requests not installed. Install with: pip install requests")

# Progress Bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    if _SHOW_WARNINGS:
        print("Warning: tqdm not installed. Install with: pip install tqdm")

# ============================================================================
# PROJECT-SPECIFIC IMPORTS
# ============================================================================

# Note: These would be imports from your own modules if you create them
# Example:
# from .image_enhancer import ImageEnhancer
# from .ppe_detector import PPEDetector
# from .false_positive_filter import FalsePositiveFilter

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Safety Equipment Classes
SAFETY_CLASSES = {
    'helmet': 0,
    'vest': 1,
    'gloves': 2,
    'safety_glasses': 3,
    'person': 4
}

# Class Names List
CLASS_NAMES = ['helmet', 'vest', 'gloves', 'safety_glasses', 'person']

# Default Model Paths
DEFAULT_MODEL_PATHS = {
    'yolov8n': 'yolov8n.pt',
    'yolov8s': 'yolov8s.pt',
    'yolov8m': 'yolov8m.pt',
    'yolov8l': 'yolov8l.pt',
    'yolov8x': 'yolov8x.pt'
}

# Default Detection Parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_IMAGE_SIZE = 640

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_imports():
    """
    Check if all critical imports are available.
    Returns dictionary with import status.
    """
    status = {
        'opencv': cv2 is not None,
        'numpy': np is not None,
        'yolo': YOLO is not None,
        'torch': torch is not None,
        'yaml': yaml is not None,
        'pillow': Image is not None,
    }
    return status

def print_import_status():
    """Print status of all imports"""
    print("=" * 70)
    print("IMPORT STATUS")
    print("=" * 70)
    status = check_imports()
    for module, available in status.items():
        symbol = "✓" if available else "✗"
        print(f"  {symbol} {module}")
    print("=" * 70)

def get_missing_imports():
    """Get list of missing critical imports"""
    status = check_imports()
    missing = [module for module, available in status.items() if not available]
    return missing

# ============================================================================
# CONVENIENCE ALIASES
# ============================================================================

# Common aliases for easier access
OpenCV = cv2
NumPy = np
PyTorch = torch
YOLOModel = YOLO

# ============================================================================
# VERSION INFORMATION
# ============================================================================

__version__ = "1.0.0"
__author__ = "Safety Kit Detection System"
__description__ = "Central import file for all project modules"

# ============================================================================
# AUTO-CHECK ON IMPORT (Optional)
# ============================================================================

# Uncomment the line below to automatically check imports when this file is imported
# print_import_status()

# ============================================================================
# MAIN BLOCK - Run when executed directly
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("All Modules - Import Status")
    print("=" * 70)
    print("\nThis file contains all imports for the Safety Kit Detection System.")
    print("Import it in your code, don't run it directly.")
    print("\nUsage:")
    print("  from all_modules import cv2, np, YOLO")
    print("  # or")
    print("  import all_modules as mod")
    print("\n" + "=" * 70)
    print_import_status()
    print("=" * 70)

# ============================================================================
# EXPORT LIST (for explicit imports)
# ============================================================================

__all__ = [
    # Standard Library
    'os', 'sys', 'subprocess', 'platform',
    'Path', 'shutil',
    'Dict', 'List', 'Tuple', 'Optional', 'Union', 'Any',
    'deque', 'defaultdict', 'Counter',
    'time', 'datetime', 'timedelta',
    'random', 'math',
    'json', 'pickle', 'yaml',
    
    # Computer Vision
    'cv2', 'np', 'Image', 'ImageEnhance', 'ImageFilter', 'PIL',
    'filters', 'exposure', 'morphology', 'measure', 'skimage',
    'imutils',
    
    # Machine Learning
    'torch', 'torchvision', 'nn', 'optim',
    'transforms', 'models', 'datasets',
    'YOLO', 'LOGGER', 'colorstr', 'YOLOModel',
    
    # Utilities
    'requests', 'RequestException', 'Timeout',
    'tqdm',
    
    # Constants
    'SAFETY_CLASSES', 'CLASS_NAMES', 'DEFAULT_MODEL_PATHS',
    'DEFAULT_CONFIDENCE_THRESHOLD', 'DEFAULT_IOU_THRESHOLD', 'DEFAULT_IMAGE_SIZE',
    
    # Functions
    'check_imports', 'print_import_status', 'get_missing_imports',
]

