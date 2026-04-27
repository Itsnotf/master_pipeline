#!/usr/bin/env python3
"""
Run only the PC6 (hand) detection from the `hand/disertasi-e` module.

This script locates the `pc6_detector` module under `hand/disertasi-e/modules`
and runs `detect_pc6()` with the repository's default hand image and model.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_DIR = REPO_ROOT / 'hand' / 'disertasi-e'

# Prefer local copy under pipeline_master/hand if available
LOCAL_HAND = Path(__file__).resolve().parent / 'hand'
if (LOCAL_HAND / 'modules').exists():
    sys.path.insert(0, str(LOCAL_HAND / 'modules'))

# Fallback to original repository modules
sys.path.insert(0, str(HAND_DIR / 'modules'))

try:
    from pc6_detector import detect_pc6
except Exception as e:
    print(f"[HAND_ONLY] Failed to import pc6_detector: {e}")
    raise


def find_image_path(local_dir, fallback_dir):
    """Find image file with flexible extension (.jpeg or .jpg)"""
    for ext in ['hand.jpeg', 'hand.jpg']:
        local_path = local_dir / 'images' / ext
        if local_path.exists():
            return local_path
    
    # Fallback to original directory
    for ext in ['hand.jpeg', 'hand.jpg']:
        fallback_path = fallback_dir / 'images' / ext
        if fallback_path.exists():
            return fallback_path
    
    # Default to .jpeg if nothing found
    return fallback_dir / 'images' / 'hand.jpeg'


def main():
    # Prefer local model/image if available
    model_path = (LOCAL_HAND / 'model' / 'hand_landmarker.task') if (LOCAL_HAND / 'model' / 'hand_landmarker.task').exists() else (HAND_DIR / 'model' / 'hand_landmarker.task')
    image_path = find_image_path(LOCAL_HAND, HAND_DIR)
    output_path = (LOCAL_HAND / 'output_pc6.jpg') if LOCAL_HAND.exists() else (HAND_DIR / 'output_pc6.jpg')

    print(f"[HAND_ONLY] Running PC6 detection: image={image_path} model={model_path} output={output_path}")
    res = detect_pc6(str(image_path), str(model_path), str(output_path))
    if res:
        print("[HAND_ONLY] PC6 detection finished successfully")
        return 0
    else:
        print("[HAND_ONLY] PC6 detection failed")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
