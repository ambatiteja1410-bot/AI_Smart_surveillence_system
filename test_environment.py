import cv2
import numpy as np
import pandas as pd
import matplotlib
import torch
from ultralytics import YOLO

print("=" * 40)
print("AI Smart Surveillance System")
print("=" * 40)

print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)
print("Pandas Version:", pd.__version__)
print("Matplotlib Version:", matplotlib.__version__)
print("PyTorch Version:", torch.__version__)

model = YOLO("yolov8n.pt")

print("\n✅ YOLO Model Loaded Successfully!")
print("\n🎉 Everything is working correctly!")