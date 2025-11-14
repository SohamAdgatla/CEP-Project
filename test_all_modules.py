"""
Test script for all_modules.py
Verifies that all imports work correctly
"""

import sys

def test_import():
    """Test importing all_modules"""
    try:
        import all_modules as mod
        print("✓ Successfully imported all_modules")
        return True
    except Exception as e:
        print(f"✗ Failed to import all_modules: {e}")
        return False

def test_core_modules():
    """Test core modules are available"""
    try:
        from all_modules import cv2, np, YOLO, Path
        print("✓ Core modules imported successfully")
        
        # Check if they're not None
        if cv2 is None:
            print("  ⚠ OpenCV is None (not installed)")
        if np is None:
            print("  ⚠ NumPy is None (not installed)")
        if YOLO is None:
            print("  ⚠ YOLO is None (not installed)")
        if Path is None:
            print("  ⚠ Path is None (should not happen)")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import core modules: {e}")
        return False

def test_constants():
    """Test constants are available"""
    try:
        from all_modules import SAFETY_CLASSES, CLASS_NAMES, DEFAULT_MODEL_PATHS
        print("✓ Constants imported successfully")
        print(f"  Safety classes: {len(SAFETY_CLASSES)}")
        print(f"  Class names: {len(CLASS_NAMES)}")
        print(f"  Model paths: {len(DEFAULT_MODEL_PATHS)}")
        return True
    except Exception as e:
        print(f"✗ Failed to import constants: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    try:
        from all_modules import check_imports, print_import_status, get_missing_imports
        print("✓ Utility functions imported successfully")
        
        # Test check_imports
        status = check_imports()
        print(f"  Import check: {len(status)} modules checked")
        
        # Test get_missing_imports
        missing = get_missing_imports()
        if missing:
            print(f"  ⚠ Missing modules: {', '.join(missing)}")
        else:
            print("  ✓ All critical modules available")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import utility functions: {e}")
        return False

def main():
    print("=" * 70)
    print("TESTING all_modules.py")
    print("=" * 70)
    
    tests = [
        ("Import Test", test_import),
        ("Core Modules", test_core_modules),
        ("Constants", test_constants),
        ("Utility Functions", test_utility_functions),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    print("=" * 70)
    
    if all_passed:
        print("\n✅ all_modules.py is working correctly!")
        print("\nYou can now use it in your code:")
        print("  from all_modules import cv2, np, YOLO")
        print("  # or")
        print("  import all_modules as mod")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


