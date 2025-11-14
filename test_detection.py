"""
Quick Test Script - Verify Detection System is Ready
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    try:
        import cv2
        print("  ✓ OpenCV")
    except ImportError:
        print("  ✗ OpenCV not installed")
        return False
    
    try:
        import numpy
        print("  ✓ NumPy")
    except ImportError:
        print("  ✗ NumPy not installed")
        return False
    
    try:
        from ultralytics import YOLO
        print("  ✓ Ultralytics YOLO")
    except ImportError:
        print("  ✗ Ultralytics not installed")
        return False
    
    return True

def test_camera():
    """Test if camera is accessible"""
    print("\nTesting camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ Camera working (Resolution: {frame.shape[1]}x{frame.shape[0]})")
                cap.release()
                return True
            else:
                print("  ✗ Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print("  ✗ Camera not accessible")
            return False
    except Exception as e:
        print(f"  ✗ Camera error: {e}")
        return False

def test_yolo_model():
    """Test if YOLO model can be loaded"""
    print("\nTesting YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("  ✓ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ YOLO model error: {e}")
        return False

def test_detection_scripts():
    """Test if detection scripts exist"""
    print("\nTesting detection scripts...")
    scripts = {
        'hybrid_detector.py': 'Ready for immediate use',
        'improved_detector.py': 'Requires trained model',
        'safety_detection.py': 'Requires trained model'
    }
    
    all_exist = True
    for script, status in scripts.items():
        if os.path.exists(script):
            print(f"  ✓ {script} - {status}")
        else:
            print(f"  ✗ {script} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 70)
    print("DETECTION SYSTEM READINESS TEST")
    print("=" * 70)
    
    results = {
        'Imports': test_imports(),
        'Camera': test_camera(),
        'YOLO Model': test_yolo_model(),
        'Scripts': test_detection_scripts()
    }
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test}")
    
    print("=" * 70)
    
    if all_passed:
        print("\n✅ SYSTEM IS READY FOR DETECTION!")
        print("\nTo start detection, run:")
        print("  python hybrid_detector.py")
        print("\nThis will work immediately with basic accuracy.")
        print("For maximum accuracy, train a model first (see TRAINING_GUIDE.md)")
    else:
        print("\n⚠️ SYSTEM NOT FULLY READY")
        print("\nPlease fix the issues above:")
        if not results['Imports']:
            print("  - Install dependencies: python setup.py")
        if not results['Camera']:
            print("  - Check camera connection and permissions")
        if not results['YOLO Model']:
            print("  - YOLO model will download automatically on first use")
        if not results['Scripts']:
            print("  - Detection scripts are missing")
    
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

