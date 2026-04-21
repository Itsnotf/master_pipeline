#!/usr/bin/env python3
"""
Lightweight ear inference wrapper that uses a local YOLO model at `ear/model/best.pt`.
Saves visualizations to `runs/predict/single_inference` under the ear folder.
"""

from pathlib import Path
import sys
import argparse

try:
    from ultralytics import YOLO
except Exception as e:
    print(f"[EAR] ultralytics not available: {e}")
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Run ear inference using local model")
    parser.add_argument('image', nargs='?', help='Image file to run inference on')
    parser.add_argument('--conf', type=float, default=0.05, help='Confidence threshold')
    args = parser.parse_args()

    base = Path(__file__).parent

    # Look for model in multiple likely locations (pipeline_master/ear/model, pipeline_master/model, etc.)
    candidate_models = [
        base / 'ear' / 'model' / 'best.pt',
        base / 'model' / 'best.pt',
        base / 'ear' / 'best.pt',
        base / 'best.pt',
    ]
    model_path = next((p for p in candidate_models if p.exists()), None)

    if model_path is None:
        print(f"[EAR] Local model not found in any of: {candidate_models}")
        print("[EAR] Please run pipeline_master/bootstrap_selfcontained.py to copy the model into place.")
        return 2

    # Choose input image
    if args.image:
        image_path = Path(args.image)
    else:
        # Try common local locations
        candidates = [
            base / 'images' / 'ear.jpg',
            base / 'images' / 'ear.jpeg',
        ]
        image_path = next((c for c in candidates if c.exists()), None)

    if image_path is None or not image_path.exists():
        print("[EAR] No image provided and no default image found in ear/images/")
        return 1

    print('\n' + '='*70)
    print('YOLOv8 Ear Detection (local)')
    print('='*70)
    print(f'Image: {image_path}')
    print(f'Model: {model_path}')
    print('='*70 + '\n')

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        save=True,
        project='runs',
        name='single_inference',
        imgsz=640,
    )

    total_detections = 0
    for result in results:
        try:
            boxes = result.boxes
            total_detections = len(boxes)
        except Exception:
            total_detections = 0

    print('\n' + '='*70)
    print(f'Results saved to: {base / "runs" / "predict" / "single_inference"}')
    print('='*70 + '\n')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
