"""
Setup script to install dependencies and run fraud detection analysis
"""
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--user", "--quiet"])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def check_package(package):
    """Check if a package is installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

# List of required packages
packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'sklearn': 'scikit-learn',
    'xgboost': 'xgboost',
    'lightgbm': 'lightgbm',
    'catboost': 'catboost'
}

print("="*60)
print("Checking and installing required packages...")
print("="*60)

# Check and install packages
for module_name, package_name in packages.items():
    if check_package(module_name):
        print(f"✓ {package_name} is already installed")
    else:
        print(f"Installing {package_name}...")
        install_package(package_name)

print("\n" + "="*60)
print("All packages checked. Running fraud detection model...")
print("="*60 + "\n")

# Now run the main script
try:
    exec(open('fraud_detection_model.py').read())
except Exception as e:
    print(f"Error running script: {e}")
    import traceback
    traceback.print_exc()

