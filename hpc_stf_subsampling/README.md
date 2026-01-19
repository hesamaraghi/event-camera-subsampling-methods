# HPC Spatiotemporal Filtering Subsampling

This module provides tools for batch processing large event camera datasets using spatiotemporal filtering subsampling on HPC clusters with SLURM.

## Overview

The system consists of:

1. **`prepare_batches.py`** - Scans input dataset, creates batch files, and generates SLURM submission script
2. **`worker.py`** - Processes a single batch of files (called by SLURM jobs)
3. **`check_status.py`** - Monitors job progress
4. **`resume_failed.py`** - Creates script to retry failed batches

## Quick Start

### 1. Prepare Batches

```bash
cd /data/event-camera-subsampling-methods
source .venv/bin/activate

python hpc_stf_subsampling/prepare_batches.py \
    --input_dir /path/to/bias_3 \
    --output_prefix bias_3_stf \
    --batch_size 100 \
    --tau 30 \
    --filter_size 7 \
    --sampling_threshold 0.5 \
    --image_height 480 \
    --image_width 640
```

This creates:
- Batch files in `hpc_stf_subsampling/batches/bias_3_stf_tau_30_fs_7/`
- SLURM submission script `submit_jobs.sh`
- Config and status tracking files

### 2. Submit Jobs

```bash
sbatch hpc_stf_subsampling/batches/bias_3_stf_tau_30_fs_7/submit_jobs.sh
```

### 3. Monitor Progress

```bash
python hpc_stf_subsampling/check_status.py \
    --batch_dir hpc_stf_subsampling/batches/bias_3_stf_tau_30_fs_7
```

### 4. Resume Failed Jobs (if needed)

```bash
python hpc_stf_subsampling/resume_failed.py \
    --batch_dir hpc_stf_subsampling/batches/bias_3_stf_tau_30_fs_7

sbatch hpc_stf_subsampling/batches/bias_3_stf_tau_30_fs_7/resume_failed.sh
```

## Output Structure

For input:
```
bias_3/
├── train/
│   ├── file1.h5
│   └── file2.h5
├── test/
│   └── file3.h5
└── val/
    └── file4.h5
```

Output will be:
```
bias_3_stf_tau_30_fs_7/
├── train/
│   ├── file1.h5
│   └── file2.h5
├── test/
│   └── file3.h5
└── val/
    └── file4.h5
```

## Parameters

### Subsampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tau` | 30 | Temporal constant in milliseconds |
| `--filter_size` | 7 | Spatial filter size (must be odd) |
| `--sampling_threshold` | 0.5 | Probability threshold for keeping events |
| `--image_height` | 480 | Image height in pixels |
| `--image_width` | 640 | Image width in pixels |

### Batch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 100 | Number of files per SLURM job |

### SLURM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--time_limit` | 4:00:00 | Time limit per job |
| `--memory` | 16G | Memory per job |
| `--cpus` | 1 | CPUs per task |
| `--partition` | batch | SLURM partition |

## Resume Capability

The system is fully resumable:

1. **Per-file tracking**: Each processed file is logged, so interrupted jobs resume from where they left off
2. **Batch status**: Completed/failed/in-progress batches are tracked
3. **Output file check**: Worker skips files that already exist in output directory

If an HPC job fails or times out:
1. Run `check_status.py` to see which batches failed
2. Run `resume_failed.py` to create a new submission script
3. Submit the resume script

## Files Created

```
hpc_stf_subsampling/batches/bias_3_stf_tau_30_fs_7/
├── batch_0000.json          # Batch file list
├── batch_0001.json
├── ...
├── config.json              # Processing parameters
├── status.json              # Global status tracking
├── submit_jobs.sh           # SLURM submission script
├── resume_failed.sh         # Resume script (created on demand)
├── logs/                    # SLURM job logs
│   ├── job_12345_0.out
│   └── job_12345_0.err
└── progress/                # Per-batch progress files
    ├── batch_0000_progress.json
    └── ...
```

## Troubleshooting

### Jobs Timing Out

Increase time limit or reduce batch size:
```bash
python prepare_batches.py --batch_size 50 --time_limit 8:00:00 ...
```

### Memory Issues

Increase memory allocation:
```bash
python prepare_batches.py --memory 32G ...
```
