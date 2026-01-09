import argparse
from ultralytics import YOLO
from config import Config

def evaluate(checkpoint):
    print(f"Loading {checkpoint}...")
    model = YOLO(checkpoint)
    results = model.val(
        data=Config.data_yaml,
        imgsz=Config.image_size,
        batch=1,
        device=Config.device,
        split="test",
    )
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"mAP@0.5:      {results.box.map50:.4f}")
    print(f"mAP@0.75:     {results.box.map75:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print("=" * 50)
    print("\nPer-class AP@0.5:")

    for i, ap in enumerate(results.box.ap50):
        class_name = results.names[i]
        print(f"  {class_name:20s}: {ap:.3f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/train/weights/best.pt")
    args = parser.parse_args()
    
    evaluate(args.checkpoint)
