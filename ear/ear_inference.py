#!/usr/bin/env python3
"""
Wrapper that delegates to pipeline_master/ear_inference.py
Placed here so run_pipeline can detect a local ear/ wrapper under pipeline_master/ear.
"""
from pathlib import Path
import sys

# Ensure pipeline_master is on path so we can import ear_inference module
base = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base))

try:
    from ear_inference import main
except Exception as e:
    print(f"[EAR WRAPPER] failed to import ear_inference: {e}")
    raise


if __name__ == '__main__':
    raise SystemExit(main())
