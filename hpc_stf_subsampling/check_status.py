#!/usr/bin/env python3
"""
Check status of HPC spatiotemporal filtering subsampling jobs.

Usage:
    python check_status.py --batch_dir /path/to/batches
"""

import argparse
import json
from pathlib import Path


def check_status(batch_dir: Path):
    """Check and display job status by aggregating individual batch status files."""
    config_file = batch_dir / 'config.json'
    
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return 1
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    total = config['num_batches']
    
    # Aggregate status from individual batch status files
    batch_status_dir = batch_dir / 'batch_status'
    progress_dir = batch_dir / 'progress'
    
    completed_batches = []
    failed_batches = []
    in_progress = []
    
    for batch_id in range(total):
        # Check batch status file
        status_file = batch_status_dir / f'batch_{batch_id:04d}.json' if batch_status_dir.exists() else None
        progress_file = progress_dir / f'batch_{batch_id:04d}_progress.json' if progress_dir.exists() else None
        
        if status_file and status_file.exists():
            with open(status_file, 'r') as f:
                batch_status = json.load(f)
            if batch_status['status'] == 'complete':
                completed_batches.append(batch_id)
            elif batch_status['status'] == 'fail':
                failed_batches.append(batch_id)
            elif batch_status['status'] == 'start':
                in_progress.append(batch_id)
        elif progress_file and progress_file.exists():
            # Has progress but no final status - likely in progress or interrupted
            in_progress.append(batch_id)
    
    completed = len(completed_batches)
    failed = len(failed_batches)
    in_prog = len(in_progress)
    pending = total - completed - failed - in_prog
    
    print("="*60)
    print("HPC Spatiotemporal Filtering Subsampling - Status")
    print("="*60)
    print(f"\nBatch directory: {batch_dir}")
    print(f"Output prefix: {config['output_prefix']}")
    print(f"Parameters: tau={config['tau']}ms, filter_size={config['filter_size']}")
    print()
    print(f"Total batches:    {total}")
    print(f"Completed:        {completed} ({completed/total*100:.1f}%)")
    print(f"Failed:           {failed}")
    print(f"In progress:      {in_prog}")
    print(f"Pending:          {pending}")
    print()
    
    # Progress bar
    bar_width = 50
    progress = completed / total
    filled = int(bar_width * progress)
    bar = '█' * filled + '░' * (bar_width - filled)
    print(f"Progress: [{bar}] {progress*100:.1f}%")
    print()
    
    # Show failed batches
    if failed_batches:
        print("Failed batches:")
        for batch_id in sorted(failed_batches):
            progress_file = batch_dir / 'progress' / f'batch_{batch_id:04d}_progress.json'
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                n_failed = len(progress_data.get('failed', []))
                n_completed = len(progress_data.get('completed', []))
                print(f"  Batch {batch_id}: {n_failed} failed, {n_completed} completed")
            else:
                print(f"  Batch {batch_id}")
        print()
    
    # Count total files processed
    total_files_done = 0
    batches_dir = batch_dir / 'batches'
    for batch_id in completed_batches:
        batch_file = batches_dir / f'batch_{batch_id:04d}.json'
        if batch_file.exists():
            with open(batch_file, 'r') as f:
                batch = json.load(f)
            total_files_done += len(batch['files'])
    
    # Also count from progress files
    progress_dir = batch_dir / 'progress'
    if progress_dir.exists():
        for progress_file in progress_dir.glob('batch_*_progress.json'):
            batch_id = int(progress_file.stem.split('_')[1])
            if batch_id not in completed_batches:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                total_files_done += len(progress_data.get('completed', []))
    
    print(f"Total files processed: ~{total_files_done}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Check status of HPC spatiotemporal filtering subsampling jobs'
    )
    parser.add_argument('--batch_dir', type=str, required=True,
                        help='Path to batch directory')
    
    args = parser.parse_args()
    return check_status(Path(args.batch_dir))


if __name__ == '__main__':
    exit(main())
