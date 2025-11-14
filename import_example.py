"""
Example: How to use all_modules.py
===================================
This file demonstrates different ways to import and use all_modules.py
"""

# ============================================================================
# METHOD 1: Import everything (Simple but not recommended for large projects)
# ============================================================================
# from all_modules import *
# 
# # Now you can use modules directly:
# frame = cv2.imread('image.jpg')
# result = np.array([1, 2, 3])
# model = YOLO('yolov8n.pt')

# ============================================================================
# METHOD 2: Import as a module (Recommended)
# ============================================================================
import all_modules as mod

# Use with module prefix:
# frame = mod.cv2.imread('image.jpg')
# result = mod.np.array([1, 2, 3])
# model = mod.YOLO('yolov8n.pt')

# ============================================================================
# METHOD 3: Import specific items (Best practice)
# ============================================================================
from all_modules import (
    cv2, np, YOLO, torch,
    Path, Dict, List, Tuple,
    time, os
)

# Use directly:
# frame = cv2.imread('image.jpg')
# result = np.array([1, 2, 3])
# model = YOLO('yolov8n.pt')

# ============================================================================
# METHOD 4: Check imports before using
# ============================================================================
from all_modules import check_imports, print_import_status

# Check what's available
status = check_imports()
if status['opencv'] and status['numpy']:
    print("Core modules available!")
else:
    print("Some modules missing!")

# Print full status
print_import_status()

# ============================================================================
# EXAMPLE: Using in your detection code
# ============================================================================

def example_detection():
    """Example of using all_modules in detection code"""
    from all_modules import cv2, np, YOLO, Path, time
    
    # Check if modules are available
    if cv2 is None or np is None or YOLO is None:
        print("Required modules not available!")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Load model
    model = YOLO('yolov8n.pt')
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame)
        
        # Display
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Import Examples")
    print("=" * 70)
    print("\nThis file shows different ways to use all_modules.py")
    print("\nUncomment the method you prefer in the code above.")
    print("\nRecommended: Use Method 3 (specific imports)")

