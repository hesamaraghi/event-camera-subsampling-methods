#!/usr/bin/env python3
"""
Resume failed HPC spatiotemporal filtering subsampling jobs.

This script creates a new SLURM job array for only the failed batches.

Usage:
    python resume_failed.py --batch_dir /path/to/batches
"""

import argparse
import json
import os
from pathlib import Path


def resume_failed(batch_dir: Path):
    """Create resume script for failed batches."""
    config_file = batch_dir / 'config.json'
    
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return 1
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    total = config['num_batches']
    
    # Find failed batches by scanning batch status files
    batch_status_dir = batch_dir / 'batch_status'
    failed_batches = []
    
    for batch_id in range(total):
        status_file = batch_status_dir / f'batch_{batch_id:04d}.json' if batch_status_dir.exists() else None
        if status_file and status_file.exists():
            with open(status_file, 'r') as f:
                batch_status = json.load(f)
            if batch_status['status'] == 'fail':
                failed_batches.append(batch_id)
    
    failed_batches = sorted(failed_batches)
    
    if not failed_batches:
        print("No failed batches to resume!")
        return 0
    
    print(f"Found {len(failed_batches)} failed batches: {failed_batches}")
    
    # Create array specification for specific batch IDs
    if len(failed_batches) == 1:
        array_spec = str(failed_batches[0])
    else:
        # Group consecutive numbers
        ranges = []
        start = failed_batches[0]
        end = failed_batches[0]
        
        for batch_id in failed_batches[1:]:
            if batch_id == end + 1:
                end = batch_id
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = batch_id
                end = batch_id
        
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        array_spec = ','.join(ranges)
    
    # Read original submit script to get SLURM settings
    original_script = batch_dir / 'submit_jobs.sh'
    with open(original_script, 'r') as f:
        original_content = f.read()
    
    # Extract SLURM settings
    import re
    time_match = re.search(r'#SBATCH --time=(\S+)', original_content)
    mem_match = re.search(r'#SBATCH --mem=(\S+)', original_content)
    cpu_match = re.search(r'#SBATCH --cpus-per-task=(\d+)', original_content)
    partition_match = re.search(r'#SBATCH --partition=(\S+)', original_content)
    
    time_limit = time_match.group(1) if time_match else '4:00:00'
    memory = mem_match.group(1) if mem_match else '16G'
    cpus = cpu_match.group(1) if cpu_match else '1'
    partition = partition_match.group(1) if partition_match else 'batch'
    
    # Extract project directory
    project_dir_match = re.search(r'cd (\S+)', original_content)
    project_dir = project_dir_match.group(1) if project_dir_match else '/data/event-camera-subsampling-methods'
    
    # Create resume script
    resume_script = f'''#!/bin/bash
#SBATCH --job-name=stf_resume
#SBATCH --output={batch_dir}/logs/resume_%A_%a.out
#SBATCH --error={batch_dir}/logs/resume_%A_%a.err
#SBATCH --array={array_spec}
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}

# Resume failed spatiotemporal filtering jobs

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
    
    resume_script_file = batch_dir / 'resume_failed.sh'
    with open(resume_script_file, 'w') as f:
        f.write(resume_script)
    
    try:
        os.chmod(resume_script_file, 0o755)
    except (PermissionError, OSError):
        pass  # Network filesystems may not support chmod
    
    # Remove old batch status files for failed batches so they can be reprocessed
    if batch_status_dir.exists():
        for batch_id in failed_batches:
            status_file = batch_status_dir / f'batch_{batch_id:04d}.json'
            if status_file.exists():
                status_file.unlink()
    
    print(f"\nCreated resume script: {resume_script_file}")
    print(f"\nTo resume failed batches, run:")
    print(f"  sbatch {resume_script_file}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Resume failed HPC spatiotemporal filtering subsampling jobs'
    )
    parser.add_argument('--batch_dir', type=str, required=True,
                        help='Path to batch directory')
    
    args = parser.parse_args()
    return resume_failed(Path(args.batch_dir))


if __name__ == '__main__':
    exit(main())
