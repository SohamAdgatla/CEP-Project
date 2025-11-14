"""
Setup script for Safety Kit Detection System
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Error installing requirements")
        return False

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera detected")
            cap.release()
            return True
        else:
            print("⚠ Camera not detected (may work when you run the application)")
            return False
    except ImportError:
        print("⚠ OpenCV not installed yet (will be installed with requirements)")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("Safety Kit Detection System - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\nPlease install requirements manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Check camera
    check_camera()
    
    print("\n" + "=" * 60)
    print("Setup completed!")
    print("=" * 60)
    print("\nTo run the detection system:")
    print("  python improved_detector.py")
    print("\nFor best results, train a custom model (see README.md)")

if __name__ == "__main__":
    main()

