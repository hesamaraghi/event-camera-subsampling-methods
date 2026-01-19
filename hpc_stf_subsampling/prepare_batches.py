#!/usr/bin/env python3
"""
Prepare batch files for HPC spatiotemporal filtering subsampling.

This script:
1. Scans the input dataset directory for all .h5 files
2. Creates batch files with file paths (configurable batch size)
3. Generates a SLURM array job submission script

Usage:
    python prepare_batches.py --input_dir /path/to/bias_3 \
                              --output_prefix stf_subsampled \
                              --batch_size 100 \
                              --tau 30 \
                              --filter_size 7 \
                              --sampling_threshold 0.5
"""

import argparse
import os
from pathlib import Path
import json


def find_all_h5_files(input_dir: Path) -> dict[str, list[Path]]:
    """
    Find all .h5 files in train/test/val subdirectories.
    
    Returns:
        Dictionary mapping split name to list of file paths
    """
    splits = ['train', 'test', 'val']
    files_by_split = {}
    
    for split in splits:
        split_dir = input_dir / split
        if split_dir.exists():
            h5_files = sorted(split_dir.glob('*.h5'))
            if h5_files:
                files_by_split[split] = h5_files
                print(f"Found {len(h5_files)} files in {split}/")
    
    return files_by_split


def create_batch_files(files_by_split: dict[str, list[Path]], 
                       batch_dir: Path, 
                       batch_size: int) -> list[Path]:
    """
    Create batch files containing file paths.
    
    Each batch file is a JSON with:
    - batch_id
    - files: list of {input_path, output_path, split}
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten all files with their split info
    all_files = []
    for split, files in files_by_split.items():
        for f in files:
            all_files.append({
                'input_path': str(f),
                'split': split,
                'filename': f.name
            })
    
    print(f"\nTotal files to process: {len(all_files)}")
    
    # Create batches folder
    batches_folder = batch_dir / 'batches'
    batches_folder.mkdir(exist_ok=True)
    
    # Create batches
    batch_files = []
    for batch_idx, start_idx in enumerate(range(0, len(all_files), batch_size)):
        batch_data = {
            'batch_id': batch_idx,
            'files': all_files[start_idx:start_idx + batch_size]
        }
        
        batch_file = batches_folder / f'batch_{batch_idx:04d}.json'
        with open(batch_file, 'w') as f:
            json.dump(batch_data, f, indent=2)
        
        batch_files.append(batch_file)
    
    print(f"Created {len(batch_files)} batch files in {batches_folder} (batch_size={batch_size})")
    return batch_files


def create_config_file(args, batch_dir: Path, num_batches: int):
    """Create a config file with all parameters for the worker script."""
    config = {
        'input_dir': str(args.input_dir),
        'output_prefix': args.output_prefix,
        'tau': args.tau,
        'filter_size': args.filter_size,
        'sampling_threshold': args.sampling_threshold,
        'image_height': args.image_height,
        'image_width': args.image_width,
        'num_batches': num_batches,
        'batch_dir': str(batch_dir),
        'batches_dir': str(batch_dir / 'batches')
    }
    
    config_file = batch_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file: {config_file}")
    return config_file


def create_slurm_script(batch_dir: Path, num_batches: int, args, project_dir: Path):
    """Create SLURM array job submission script."""
    
    # Compute output directory name
    output_dir_name = f"{args.output_prefix}_tau_{args.tau}_fs_{args.filter_size}"
    
    slurm_script = f'''#!/bin/bash
#SBATCH --job-name=stf_subsample
#SBATCH --output={batch_dir}/logs/job_%A_%a.out
#SBATCH --error={batch_dir}/logs/job_%A_%a.err
#SBATCH --array=0-{num_batches - 1}
#SBATCH --partition={args.partition}
#SBATCH --qos={args.qos}
#SBATCH --time={args.time_limit}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --mem={args.memory}
#SBATCH --mail-type=END

# Spatiotemporal Filtering Subsampling - HPC Batch Processing
# Output directory: {output_dir_name}

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export PYTHONHASHSEED="0"

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="

# Activate virtual environment
cd {project_dir}
source .venv/bin/activate

# Run worker script for this batch
python hpc_stf_subsampling/worker.py \\
    --batch_dir {batch_dir} \\
    --batch_id $SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
'''
    
    script_file = batch_dir / 'submit_jobs.sh'
    with open(script_file, 'w') as f:
        f.write(slurm_script)
    
    try:
        os.chmod(script_file, 0o755)
    except (PermissionError, OSError):
        pass  # Network filesystems may not support chmod
    print(f"Created SLURM script: {script_file}")
    
    # Create logs directory
    logs_dir = batch_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    return script_file


def create_status_tracking(batch_dir: Path, num_batches: int):
    """Create status tracking file for resume capability."""
    status = {
        'total_batches': num_batches,
        'completed_batches': [],
        'failed_batches': [],
        'in_progress': []
    }
    
    status_file = batch_dir / 'status.json'
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    print(f"Created status tracking file: {status_file}")
    return status_file


def main():
    parser = argparse.ArgumentParser(
        description='Prepare batch files for HPC spatiotemporal filtering subsampling'
    )
    
    # Input/Output paths
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to input dataset (e.g., /path/to/bias_3)')
    parser.add_argument('--output_prefix', type=str, required=True,
                        help='Prefix for output directory name (tau and filter_size will be appended)')
    
    # Batch configuration
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of files per batch (default: 100)')
    
    # Subsampling parameters
    parser.add_argument('--tau', type=int, default=30,
                        help='Temporal constant in milliseconds (default: 30)')
    parser.add_argument('--filter_size', type=int, default=7,
                        help='Spatial filter size (must be odd, default: 7)')
    parser.add_argument('--sampling_threshold', type=float, default=0.5,
                        help='Sampling threshold (default: 0.5)')
    
    # Image dimensions (optional - auto-detected from data if not provided)
    parser.add_argument('--image_height', type=int, default=0,
                        help='Image height (default: 0 = auto-detect from data)')
    parser.add_argument('--image_width', type=int, default=0,
                        help='Image width (default: 0 = auto-detect from data)')
    
    # SLURM parameters
    parser.add_argument('--time_limit', type=str, default='04:00:00',
                        help='Time limit per job (default: 04:00:00)')
    parser.add_argument('--memory', type=str, default='16000',
                        help='Memory per job in MB (default: 16000)')
    parser.add_argument('--cpus', type=int, default=2,
                        help='CPUs per task (default: 2)')
    parser.add_argument('--partition', type=str, default='prb,insy,general',
                        help='SLURM partition (default: prb,insy,general)')
    parser.add_argument('--qos', type=str, default='short',
                        help='SLURM QOS (default: short)')
    
    args = parser.parse_args()
    
    # Validate inputs
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    if args.filter_size % 2 == 0:
        print(f"Error: filter_size must be odd, got {args.filter_size}")
        return 1
    
    print("="*60)
    print("HPC Spatiotemporal Filtering Subsampling - Batch Preparation")
    print("="*60)
    print(f"\nInput directory: {input_dir}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Batch size: {args.batch_size}")
    print(f"\nSubsampling parameters:")
    print(f"  tau: {args.tau} ms")
    print(f"  filter_size: {args.filter_size}")
    print(f"  sampling_threshold: {args.sampling_threshold}")
    if args.image_width > 0 and args.image_height > 0:
        print(f"  image_size: {args.image_width}x{args.image_height}")
    else:
        print(f"  image_size: auto-detect from data")
    
    # Auto-detect project directory from script location
    project_dir = Path(__file__).parent.parent.resolve()
    
    # Create batch directory
    batch_dir = project_dir / 'hpc_stf_subsampling' / 'batches' / f'{args.output_prefix}_tau_{args.tau}_fs_{args.filter_size}'
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProject directory: {project_dir}")
    print(f"Batch directory: {batch_dir}")
    
    # Find all files
    print("\nScanning for .h5 files...")
    files_by_split = find_all_h5_files(input_dir)
    
    if not files_by_split:
        print("Error: No .h5 files found in train/test/val subdirectories")
        return 1
    
    # Create batch files
    print("\nCreating batch files...")
    batch_files = create_batch_files(files_by_split, batch_dir, args.batch_size)
    
    # Create config file
    print("\nCreating config file...")
    create_config_file(args, batch_dir, len(batch_files))
    
    # Create status tracking
    print("\nCreating status tracking file...")
    create_status_tracking(batch_dir, len(batch_files))
    
    # Create SLURM script
    print("\nCreating SLURM submission script...")
    slurm_script = create_slurm_script(batch_dir, len(batch_files), args, project_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("PREPARATION COMPLETE")
    print("="*60)
    print(f"\nTotal files: {sum(len(f) for f in files_by_split.values())}")
    print(f"Total batches: {len(batch_files)}")
    print(f"\n--- TEST A SINGLE BATCH FIRST ---")
    print(f"Run one batch locally to verify:")
    print(f"  cd {project_dir}")
    print(f"  source .venv/bin/activate")
    print(f"  python hpc_stf_subsampling/worker.py --batch_dir {batch_dir} --batch_id 0")
    print(f"\nOr submit just one SLURM job:")
    print(f"  sbatch --array=0 {slurm_script}")
    print(f"\n--- SUBMIT ALL JOBS ---")
    print(f"  sbatch {slurm_script}")
    print(f"\n--- MONITOR ---")
    print(f"  python hpc_stf_subsampling/check_status.py --batch_dir {batch_dir}")
    print(f"\n--- RESUME FAILED ---")
    print(f"  python hpc_stf_subsampling/resume_failed.py --batch_dir {batch_dir}")
    
    return 0


if __name__ == '__main__':
    exit(main())
