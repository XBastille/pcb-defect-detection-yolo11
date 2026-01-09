import torch

class Config:
    data_yaml = "./data/DsPCBSD/data.yaml"
    model_name = "yolo11l.pt"
    image_size = 1024
    batch_size = 16
    epochs = 75
    patience = 15
    device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = 4
    models_dir = "./models"
    output_dir = "./outputs"
    mosaic = 1.0
    mixup = 0.0
    copy_paste = 0.3
    degrees = 10.0
    translate = 0.1
    scale = 0.5
    fliplr = 0.5
    hsv_h = 0.015
    hsv_s = 0.7
    hsv_v = 0.4
