# All Modules Import Guide

## Overview

The `all_modules.py` file contains **all imports** used across the Safety Kit Detection System project. This centralizes all dependencies in one place for easy management.

## Quick Start

### Method 1: Import Everything (Simple)
```python
from all_modules import *

# Now use directly:
frame = cv2.imread('image.jpg')
model = YOLO('yolov8n.pt')
```

### Method 2: Import as Module (Recommended)
```python
import all_modules as mod

# Use with prefix:
frame = mod.cv2.imread('image.jpg')
model = mod.YOLO('yolov8n.pt')
```

### Method 3: Specific Imports (Best Practice)
```python
from all_modules import cv2, np, YOLO, Path, time

# Use directly:
frame = cv2.imread('image.jpg')
model = YOLO('yolov8n.pt')
```

## What's Included

### Standard Library
- `os`, `sys`, `subprocess`, `platform`
- `pathlib.Path`, `shutil`
- `typing` (Dict, List, Tuple, Optional, etc.)
- `collections` (deque, defaultdict, Counter)
- `time`, `datetime`
- `random`, `math`
- `json`, `pickle`, `yaml`

### Computer Vision
- `cv2` (OpenCV)
- `numpy` (as `np`)
- `PIL` (Pillow)
- `skimage` (scikit-image)
- `imutils`

### Machine Learning
- `torch`, `torchvision` (PyTorch)
- `ultralytics.YOLO`
- `torch.nn`, `torch.optim`

### Utilities
- `requests` (HTTP)
- `tqdm` (Progress bars)

## Checking Imports

```python
from all_modules import check_imports, print_import_status

# Check what's available
status = check_imports()
if status['opencv']:
    print("OpenCV available!")

# Print full status
print_import_status()
```

## Constants Included

```python
from all_modules import SAFETY_CLASSES, CLASS_NAMES, DEFAULT_MODEL_PATHS

# Safety equipment classes
print(SAFETY_CLASSES)  # {'helmet': 0, 'vest': 1, ...}

# Class names list
print(CLASS_NAMES)  # ['helmet', 'vest', 'gloves', ...]

# Default model paths
print(DEFAULT_MODEL_PATHS)  # {'yolov8n': 'yolov8n.pt', ...}
```

## Usage in Your Code

### Example 1: Detection Script
```python
from all_modules import cv2, np, YOLO, time

cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    results = model(frame)
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Example 2: Training Script
```python
from all_modules import YOLO, Path, yaml, torch

model = YOLO('yolov8n.pt')
results = model.train(data='dataset.yaml', epochs=100)
```

### Example 3: Data Processing
```python
from all_modules import Path, cv2, np, json

image_path = Path('images/train')
for img_file in image_path.glob('*.jpg'):
    img = cv2.imread(str(img_file))
    # Process image...
```

## Adding New Modules

When you add new modules to the project:

1. **Add to `all_modules.py`**:
   ```python
   # In the appropriate section
   try:
       import new_module
   except ImportError:
       new_module = None
       print("Warning: new_module not installed")
   ```

2. **Add to `__all__` list**:
   ```python
   __all__ = [
       # ... existing imports
       'new_module',
   ]
   ```

3. **Update `requirements.txt`** if it's a new dependency

## Error Handling

The file includes try-except blocks for all imports. If a module is missing:
- It sets the variable to `None`
- Prints a warning message
- Your code can check: `if cv2 is not None: ...`

## Benefits

✅ **Single source of truth** for all imports
✅ **Easy to manage** dependencies
✅ **Error handling** built-in
✅ **Consistent** across all files
✅ **Easy to update** when adding new modules

## Best Practices

1. **Use specific imports** when possible (Method 3)
2. **Check availability** before using optional modules
3. **Keep `all_modules.py` updated** when adding new dependencies
4. **Document new imports** in comments

## Troubleshooting

### Module not found?
- Check if it's in `all_modules.py`
- Verify it's in `requirements.txt`
- Install: `pip install -r requirements.txt`

### Import errors?
- Run: `python -c "import all_modules; all_modules.print_import_status()"`
- Check which modules are missing
- Install missing dependencies

## See Also

- `import_example.py` - Usage examples
- `requirements.txt` - All dependencies
- `setup.py` - Installation script

