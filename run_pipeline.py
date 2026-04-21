#!/usr/bin/env python3
"""
Master pipeline to run hand (PC6 only) and
ear-hunger-point.yolov8/simple_inference.py in parallel
and collect outputs into a single folder.
"""

from pathlib import Path
import subprocess
import sys
import shutil
import argparse
import concurrent.futures


REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_DIR = REPO_ROOT / 'hand' / 'disertasi-e'
EAR_DIR = REPO_ROOT / 'ear-hunger-point.yolov8'
OUT_DIR = Path(__file__).resolve().parent / 'output'

# Local runner (we add this file to call only PC6 detection)
HAND_RUNNER = Path(__file__).resolve().parent / 'hand_only.py'
EAR_SCRIPT = 'simple_inference.py'


def find_ear_image(hand_dir, ear_dir):
    candidates = [
        hand_dir / 'images' / 'ear.jpg',
        hand_dir / 'images' / 'ear.jpeg',
        ear_dir / 'images' / 'ear.jpg',
        ear_dir / 'images' / 'ear.jpeg',
        REPO_ROOT / 'ear' / 'images' / 'ear.jpg',
        REPO_ROOT / 'ear' / 'images' / 'ear.jpeg',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_process(cmd, cwd, name):
    print(f"[{name}] Running: {' '.join(cmd)} (cwd={cwd})")
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        print(f"[{name}] Exit {proc.returncode}")
        if proc.stdout:
            print(f"[{name}] STDOUT:\n{proc.stdout}")
        if proc.stderr:
            print(f"[{name}] STDERR:\n{proc.stderr}")
        return proc.returncode == 0
    except Exception as e:
        print(f"[{name}] Exception: {e}")
        return False


def copy_outputs(output_dir, hand_dir, ear_dir, include_hunger=False, include_legacy_ear=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    # Clean unwanted legacy files in output_dir unless requested
    if not include_hunger:
        for p in list(output_dir.glob('hand_output_hunger*')):
            try:
                p.unlink()
            except Exception:
                pass
    if not include_legacy_ear:
        for p in list(output_dir.glob('ear_alt_*')):
            try:
                p.unlink()
            except Exception:
                pass

    # Hand outputs (created by pc6 detector). Include hunger output only when requested.
    hand_outs = ['output_pc6.jpg']
    if include_hunger:
        hand_outs.append('output_hunger.jpg')
    for name in hand_outs:
        src = hand_dir / name
        if src.exists():
            dest = output_dir / f"hand_{name}"
            shutil.copy2(src, dest)
            copied.append(dest)

    # Ear YOLO inference outputs: search recursively for any 'single_inference' folders
    for ear_inf in ear_dir.glob('**/single_inference'):
        if ear_inf.is_dir():
            for f in ear_inf.rglob('*'):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.json'):
                    dest = output_dir / f"ear_{f.name}"
                    shutil.copy2(f, dest)
                    copied.append(dest)

    # Also include legacy ear outputs from `ear/output` if explicitly requested
    alt_ear_out = REPO_ROOT / 'ear' / 'output'
    if include_legacy_ear and alt_ear_out.exists():
        for f in alt_ear_out.glob('*.jpg'):
            dest = output_dir / f"ear_alt_{f.name}"
            shutil.copy2(f, dest)
            copied.append(dest)

    return copied


def main():
    parser = argparse.ArgumentParser(description="Run hand and ear detection scripts concurrently and gather outputs.")
    parser.add_argument('--ear-image', help="Path to ear image to feed to ear detector")
    parser.add_argument('--output', help="Output folder", default=str(OUT_DIR))
    parser.add_argument('--dry-run', action='store_true', help="Don't execute heavy commands; only print what would run")
    parser.add_argument('--include-hunger', action='store_true', help="Also include hand's output_hunger.jpg in the collected outputs")
    parser.add_argument('--include-legacy-ear', action='store_true', help="Also include legacy ear outputs from ear/output folder")
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()

    # Prefer pipeline_master local copies if they exist
    local_hand_dir = Path(__file__).resolve().parent / 'hand'
    local_ear_dir = Path(__file__).resolve().parent / 'ear'

    hand_dir = local_hand_dir if local_hand_dir.exists() else HAND_DIR
    ear_dir = local_ear_dir if local_ear_dir.exists() else EAR_DIR

    ear_image = Path(args.ear_image).expanduser().resolve() if args.ear_image else find_ear_image(hand_dir, ear_dir)
    if ear_image is None:
        print("Warning: no ear image found. The ear detector will be run without explicit image argument and may use its defaults.")

    hand_cmd = [sys.executable, str(HAND_RUNNER)]

    # Prefer local ear wrapper if present
    local_ear_wrapper = Path(__file__).resolve().parent / 'ear' / 'ear_inference.py'
    if local_ear_wrapper.exists():
        ear_cmd = [sys.executable, str(local_ear_wrapper)] + ([str(ear_image)] if ear_image else [])
        ear_cwd = local_ear_wrapper.parent
    else:
        ear_cmd = [sys.executable, EAR_SCRIPT] + ([str(ear_image)] if ear_image else [])
        ear_cwd = ear_dir

    print(f"Repository root: {REPO_ROOT}")
    print(f"Hand dir: {hand_dir}")
    print(f"Ear dir: {ear_dir}")
    print(f"Ear image: {ear_image}")
    print(f"Output dir: {output_dir}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        print("\nDRY RUN: Commands that would be executed:")
        print("  Hand:", ' '.join(map(str, hand_cmd)), f"(cwd={HAND_RUNNER.parent})")
        print("  Ear: ", ' '.join(map(str, ear_cmd)), f"(cwd={ear_cwd})")
        print("\nDry-run finished. No processes executed.")
        return 0

    # Run both processes concurrently
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_hand = ex.submit(run_process, hand_cmd, HAND_RUNNER.parent, 'HAND')
        fut_ear = ex.submit(run_process, ear_cmd, ear_cwd, 'EAR')
        results['hand_ok'] = fut_hand.result()
        results['ear_ok'] = fut_ear.result()

    print("\nCollecting outputs...")
    copied = copy_outputs(output_dir, hand_dir, ear_dir, include_hunger=args.include_hunger, include_legacy_ear=args.include_legacy_ear)
    print(f"Copied {len(copied)} files to {output_dir}")
    for p in copied:
        print(f"  - {p.name}")

    print("\nSummary:")
    print(f"  Hand detection success: {results['hand_ok']}")
    print(f"  Ear detection success: {results['ear_ok']}")
    print("Pipeline finished.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
