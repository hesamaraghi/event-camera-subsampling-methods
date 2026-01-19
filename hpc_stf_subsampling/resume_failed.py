#!/usr/bin/env python3
"""
Resume failed HPC spatiotemporal filtering subsampling jobs.

This script identifies failed batches and prints the sbatch command to rerun them.

Usage:
    python resume_failed.py --batch_dir /path/to/batches --sbatch_script /path/to/your_sbatch.sh
"""

import argparse
import json
import subprocess
from pathlib import Path


def resume_failed(batch_dir: Path, sbatch_script: Path):
    """Identify failed batches and print resume command."""
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
    
    # Remove old batch status files for failed batches so they can be reprocessed
    if batch_status_dir.exists():
        for batch_id in failed_batches:
            status_file = batch_status_dir / f'batch_{batch_id:04d}.json'
            if status_file.exists():
                status_file.unlink()
                print(f"Removed status file for batch {batch_id}")
    
    # Get worker path
    project_dir = Path(__file__).parent.parent.resolve()
    worker_path = project_dir / 'hpc_stf_subsampling' / 'worker.py'
    
    # Submit failed batches
    print(f"\nSubmitting {len(failed_batches)} failed batches...")
    for batch_id in failed_batches:
        cmd = [
            'sbatch', str(sbatch_script),
            'python', str(worker_path),
            '--batch_dir', str(batch_dir),
            '--batch_id', str(batch_id)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Batch {batch_id}: {result.stdout.strip()}")
        else:
            print(f"  Batch {batch_id}: FAILED - {result.stderr.strip()}")
    
    print("\nAll failed batches resubmitted!")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Resume failed HPC spatiotemporal filtering subsampling jobs'
    )
    parser.add_argument('--batch_dir', type=str, required=True,
                        help='Path to batch directory')
    parser.add_argument('--sbatch_script', type=str, required=True,
                        help='Path to your sbatch script')
    
    args = parser.parse_args()
    
    sbatch_script = Path(args.sbatch_script)
    if not sbatch_script.exists():
        print(f"Error: sbatch script does not exist: {sbatch_script}")
        return 1
    
    return resume_failed(Path(args.batch_dir), sbatch_script)


if __name__ == '__main__':
    exit(main())
