#!/bin/bash

mkdir -p outputs/train/weights

echo "Downloading best.pt..."
curl -L "https://github.com/XBastille/pcb-defect-detection-yolo11/releases/download/v1/best.pt" -o outputs/train/weights/best.pt

echo "Done! Weights saved to outputs/train/weights/"
