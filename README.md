# Pipeline Master

This folder contains a small master script to run the two detection pipelines in this repository in parallel and gather their output images into a single folder.

Files:
- `run_pipeline.py` — main script to run both detectors and collect outputs
- `requirements.txt` — suggested packages used by detectors (install in your environment if needed)

Quick usage:

Dry-run (no heavy processing):

```bash
python pipeline_master/run_pipeline.py --dry-run
```

Run pipeline (uses defaults, will execute both scripts):

```bash
python pipeline_master/run_pipeline.py
```

Specify a custom ear image:

```bash
python pipeline_master/run_pipeline.py --ear-image "hand/disertasi-e/images/ear.jpg"
```

Outputs are copied to `pipeline_master/output/` by default.
