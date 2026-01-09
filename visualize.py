import os
import argparse
import random
import cv2
import yaml
from collections import defaultdict
from ultralytics import YOLO
from config import Config


def get_test_images_dir():
    """Find test images directory from data.yaml"""
    with open(Config.data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    data_root = os.path.dirname(os.path.abspath(Config.data_yaml))
    test_path = data.get('test', data.get('val', 'test/images'))
    test_dir = os.path.normpath(os.path.join(data_root, test_path))
    
    if not os.path.exists(test_dir):
        alternatives = [
            os.path.join(data_root, 'test', 'images'),
            os.path.join(data_root, '..', 'test', 'images'),
        ]

        for alt in alternatives:
            if os.path.exists(alt):
                test_dir = alt
                break
    
    return test_dir


def visualize(checkpoint, output_dir, test_dir=None, num_samples=20, conf_thresh=0.5):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {checkpoint}...")
    model = YOLO(checkpoint)
    if test_dir is None:
        test_dir = get_test_images_dir()

    print(f"Test images dir: {test_dir}")
    if not os.path.exists(test_dir):
        print(f"ERROR: Directory not found: {test_dir}")
        return
    
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if len(image_files) == 0:
        print(f"No images found in {test_dir}")
        return
    
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    print(f"Generating {len(samples)} visualizations...")
    class_counts = defaultdict(int)
    
    for i, img_name in enumerate(samples):
        img_path = os.path.join(test_dir, img_name)
        results = model.predict(
            source=img_path,
            imgsz=Config.image_size,
            conf=conf_thresh,
            device=Config.device,
            save=False,
        )[0]
        
        if len(results.boxes) > 0:
            main_cls = int(results.boxes.cls[0])
            class_name = results.names[main_cls].replace(" ", "_")
            
        else:
            class_name = "No_Detection"
        
        class_counts[class_name] += 1
        count = class_counts[class_name]
        out_name = f"{class_name}_{count:03d}.jpg"
        out_path = os.path.join(output_dir, out_name)
        img = results.plot()
        cv2.imwrite(out_path, img)
        print(f"  [{i+1}/{len(samples)}] Saved: {out_name}")
    
    print(f"\nDone! Visualizations saved to {output_dir}")
    print(f"\nClass distribution:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/train/weights/best.pt")
    parser.add_argument("--output_dir", type=str, default="./outputs/visualizations")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()
    
    visualize(args.checkpoint, args.output_dir, args.test_dir, args.num_samples, args.conf)
