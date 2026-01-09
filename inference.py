import argparse
import json
import cv2
from ultralytics import YOLO
from config import Config

SEVERITY_THRESHOLDS = {"low": 0.005, "medium": 0.02}

def classify_severity(area_ratio):
    """
    Severity based on area ratio (bbox_area / image_area):
      - Low:    < 0.5%
      - Medium: 0.5% - 2%
      - High:   >= 2%
    """
    if area_ratio < SEVERITY_THRESHOLDS["low"]:
        return "Low"
    elif area_ratio < SEVERITY_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def run_inference(checkpoint, image_path, output_image=None, output_json=None, conf_thresh=0.5):
    """Run inference and extract defect coordinates with severity."""
    print(f"Loading {checkpoint}...")
    model = YOLO(checkpoint)
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    image_area = img_w * img_h
    results = model.predict(
        source=image_path,
        imgsz=Config.image_size,
        conf=conf_thresh,
        device=Config.device,
        save=False,
    )[0]
    
    defects = []
    boxes = results.boxes
    
    for i in range(len(boxes)):
        box = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        bbox_area = (x2 - x1) * (y2 - y1)
        area_ratio = bbox_area / image_area
        cls_id = int(boxes.cls[i])
        confidence = float(boxes.conf[i])
        class_name = results.names[cls_id]
        severity = classify_severity(area_ratio)
        defects.append({
            "defect_type": class_name,
            "confidence": round(confidence, 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [round(center_x, 1), round(center_y, 1)],
            "area_ratio_pct": round(area_ratio * 100, 3),
            "severity": severity
        })
    
    if output_image:
        annotated = results.plot()
        cv2.imwrite(output_image, annotated)
        print(f"Saved annotated image: {output_image}")
    
    output = {
        "image": image_path,
        "dimensions": [img_w, img_h],
        "num_defects": len(defects),
        "defects": defects
    }
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"Saved JSON: {output_json}")
    
    print(f"\n{'='*60}")
    print(f"INSPECTION RESULTS: {image_path}")
    print(f"{'='*60}")
    print(f"Image: {img_w} x {img_h} | Defects: {len(defects)}")
    print(f"Severity: Low (<0.5%) | Medium (0.5-2%) | High (>=2%)")
    print(f"{'='*60}")
    
    for i, d in enumerate(defects, 1):
        print(f"\n  [{i}] {d['defect_type']} ({d['confidence']:.1%})")
        print(f"      Center: ({d['center'][0]}, {d['center'][1]})")
        print(f"      BBox: {d['bbox']}")
        print(f"      Severity: {d['severity']} ({d['area_ratio_pct']:.2f}%)")
    
    if len(defects) == 0:
        print("\n  ✓ No defects detected - PCB is PASS")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB Quality Inspection")
    parser.add_argument("--checkpoint", type=str, default="outputs/train/weights/best.pt")
    parser.add_argument("--image", type=str, required=True, help="Input image")
    parser.add_argument("--output_image", type=str, default=None, help="Save annotated image")
    parser.add_argument("--output_json", type=str, default=None, help="Save JSON results")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()
    
    run_inference(args.checkpoint, args.image, args.output_image, args.output_json, args.conf)
