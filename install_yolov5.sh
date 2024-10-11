#!/bin/bash

# Create and activate a virtual environment
python -m venv yolov5-env
source yolov5-env/bin/activate  # For Windows: yolov5-env\Scripts\activate

# Install PyTorch
# Uncomment the desired installation command based on your system
# For GPU (CUDA 11.7, for example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# For CPU-only
# pip install torch torchvision torchaudio

# Install YOLOv5 dependencies
pip install -r requirements.txt

# Install OpenCV
pip install opencv-python

# Install Pygame
pip install pygame

echo "Installation completed. Please activate the virtual environment using 'source yolov5-env/bin/activate'"
