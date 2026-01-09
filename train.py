import os
import argparse
import shutil
from ultralytics import YOLO
from config import Config


def train(resume=False):
    os.makedirs(Config.models_dir, exist_ok=True)
    os.makedirs(f"{Config.output_dir}/weights", exist_ok=True)
    weights_dir = f"{Config.output_dir}/weights"
    ultralytics_weights = f"{Config.output_dir}/train/weights"
    
    if resume:
        if os.path.exists(f"{ultralytics_weights}/last.pt"):
            print(f"Resuming from {ultralytics_weights}/last.pt...")
            model = YOLO(f"{ultralytics_weights}/last.pt")
        elif os.path.exists(f"{weights_dir}/latest.pt"):
            print(f"Resuming from {weights_dir}/latest.pt...")
            model = YOLO(f"{weights_dir}/latest.pt")
        else:
            print("No checkpoint found! Starting fresh...")
            resume = False
    
    if not resume:
        pretrained_path = f"{Config.models_dir}/{Config.model_name}"
        if not os.path.exists(pretrained_path):
            print(f"Downloading {Config.model_name} to {Config.models_dir}...")
            model = YOLO(Config.model_name)
            if os.path.exists(Config.model_name):
                shutil.move(Config.model_name, pretrained_path)
        else:
            print(f"Loading pretrained {pretrained_path}...")
        model = YOLO(pretrained_path)
    
    results = model.train(
        data=Config.data_yaml,
        epochs=Config.epochs,
        imgsz=Config.image_size,
        batch=Config.batch_size,
        device=Config.device,
        workers=Config.workers,
        patience=Config.patience,
        project=Config.output_dir,
        name="train",
        exist_ok=True,
        save=True,
        save_period=1,
        resume=resume,
        mosaic=Config.mosaic,
        mixup=Config.mixup,
        copy_paste=Config.copy_paste,
        degrees=Config.degrees,
        translate=Config.translate,
        scale=Config.scale,
        fliplr=Config.fliplr,
        hsv_h=Config.hsv_h,
        hsv_s=Config.hsv_s,
        hsv_v=Config.hsv_v,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    
    if os.path.exists(f"{ultralytics_weights}/best.pt"):
        shutil.copy(f"{ultralytics_weights}/best.pt", f"{weights_dir}/best.pt")
        print(f"Saved best model to {weights_dir}/best.pt")
    
    if os.path.exists(f"{ultralytics_weights}/last.pt"):
        shutil.copy(f"{ultralytics_weights}/last.pt", f"{weights_dir}/latest.pt")
        print(f"Saved latest model to {weights_dir}/latest.pt")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training")
    args = parser.parse_args()
    
    train(resume=args.resume)
