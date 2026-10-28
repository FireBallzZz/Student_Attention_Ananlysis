import sys
import os
import platform
import shutil
import torch

print("=== Python & OS ===")
print("Python:", sys.version)
print("Platform:", platform.platform())

print("\n=== Disk Space ===")
total, used, free = shutil.disk_usage(".")
print(f"Disk Total: {total / (1024**3):.2f} GB")
print(f"Disk Used:  {used / (1024**3):.2f} GB")
print(f"Disk Free:  {free / (1024**3):.2f} GB")

print("\n=== GPU & PyTorch ===")
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"Device VRAM: {props.total_memory / (1024**3):.2f} GB")
    print("CUDA Capability:", f"{props.major}.{props.minor}")

print("\n=== Installed Libraries ===")
packages = [
    "torch", "torchvision", "ultralytics", "sklearn", "scipy", "numpy", 
    "pandas", "cv2", "lightgbm", "xgboost", "matplotlib", "seaborn", 
    "yaml", "tqdm", "requests", "PIL"
]
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"  {pkg:15s}: {ver}")
    except ImportError:
        print(f"  {pkg:15s}: NOT INSTALLED")
