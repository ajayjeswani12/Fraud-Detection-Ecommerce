"""Test script to verify environment and packages"""
import sys

print("Python version:", sys.version)
print("\nTesting package imports...")

packages_to_test = [
    ('pandas', 'pd'),
    ('numpy', 'np'),
    ('matplotlib.pyplot', 'plt'),
    ('seaborn', 'sns'),
    ('sklearn', 'sklearn'),
    ('xgboost', 'xgb'),
    ('lightgbm', 'lgb'),
    ('catboost', 'cb')
]

failed = []
for package, alias in packages_to_test:
    try:
        __import__(package)
        print(f"✓ {package}")
    except ImportError as e:
        print(f"✗ {package} - NOT INSTALLED")
        failed.append(package)

if failed:
    print(f"\n❌ Missing packages: {', '.join(failed)}")
    print("\nPlease install missing packages using:")
    print("pip install " + " ".join(failed))
    sys.exit(1)
else:
    print("\n✓ All packages are installed!")
    print("You can now run: python fraud_detection_model.py")

