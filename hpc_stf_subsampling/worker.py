#!/usr/bin/env python3
"""
Worker script for HPC spatiotemporal filtering subsampling.

This script processes a single batch of files, applying spatiotemporal
filtering subsampling and saving results.

Features:
- Resumable: skips already processed files
- Tracks progress in status file
- Saves output in same format as input

Usage:
    python worker.py --batch_dir /path/to/batches --batch_id 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
import traceback
import time

import h5py
import hdf5plugin
import numpy as np
import torch
from torch_geometric.data import Data
from omegaconf import DictConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datatransforms.event_transforms import SpatioTemporalFilteringSubsampling
from utils.data_utils import numpy2pyg_event_convertor, pyg2numpy_event_convertor
from tonic.io import make_structured_array


def load_config(batch_dir: Path) -> dict:
    """Load configuration from batch directory."""
    config_file = batch_dir / 'config.json'
    with open(config_file, 'r') as f:
        return json.load(f)


def load_batch(batch_dir: Path, batch_id: int) -> dict:
    """Load batch file."""
    batch_file = batch_dir / f'batch_{batch_id:04d}.json'
    with open(batch_file, 'r') as f:
        return json.load(f)


def update_status(batch_dir: Path, batch_id: int, status_type: str, file_info: dict = None):
    """
    Update batch status by writing to individual batch status file.
    No locking needed - each batch writes only its own file.
    
    status_type: 'start', 'complete', 'fail'
    """
    batch_status_dir = batch_dir / 'batch_status'
    batch_status_dir.mkdir(exist_ok=True)
    
    status_file = batch_status_dir / f'batch_{batch_id:04d}.json'
    
    status_data = {
        'batch_id': batch_id,
        'status': status_type,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)


def get_processed_files(batch_dir: Path, batch_id: int) -> set:
    """Get set of already processed files for resume capability."""
    progress_file = batch_dir / 'progress' / f'batch_{batch_id:04d}_progress.json'
    
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        return set(progress.get('completed', []))
    
    return set()


def save_file_progress(batch_dir: Path, batch_id: int, filename: str):
    """Save progress for individual file (for resume). Each batch has its own file."""
    progress_dir = batch_dir / 'progress'
    progress_dir.mkdir(exist_ok=True)
    
    progress_file = progress_dir / f'batch_{batch_id:04d}_progress.json'
    
    # Read existing progress (only this batch writes to this file)
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {'completed': [], 'failed': []}
    
    if filename not in progress['completed']:
        progress['completed'].append(filename)
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def save_failed_file(batch_dir: Path, batch_id: int, filename: str, error: str):
    """Save failed file info for debugging. Each batch has its own file."""
    progress_dir = batch_dir / 'progress'
    progress_dir.mkdir(exist_ok=True)
    
    progress_file = progress_dir / f'batch_{batch_id:04d}_progress.json'
    
    # Read existing progress (only this batch writes to this file)
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {'completed': [], 'failed': []}
    
    progress['failed'].append({'file': filename, 'error': error})
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def load_events_from_h5(h5_path: Path) -> tuple[Data, dict]:
    """
    Load events from HDF5 file and convert to PyG Data.
    
    Returns:
        Tuple of (PyG Data, metadata dict)
    """
    with h5py.File(h5_path, 'r') as f:
        # Try different HDF5 structures
        if 'events' in f:
            events_group = f['events']
            if isinstance(events_group, h5py.Dataset):
                # Structured array format
                events_data = events_group[()]
                x = events_data['x']
                y = events_data['y']
                t = events_data['t'].astype(np.int64)
                p = events_data['p'].astype(bool)
            else:
                # Separate datasets format
                x = events_group['x'][()]
                y = events_group['y'][()]
                t = events_group['t'][()].astype(np.int64)
                p = events_group['p'][()].astype(bool)
        else:
            # Try root level
            x = f['x'][()]
            y = f['y'][()]
            t = f['t'][()].astype(np.int64)
            p = f['p'][()].astype(bool)
        
        # Get metadata if available
        metadata = {}
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]
    
    # Create structured array
    events_struct = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)])
    events_array = make_structured_array(x, y, t, p, dtype=events_struct)
    
    # Convert to PyG Data
    data = numpy2pyg_event_convertor(events_array)
    
    return data, metadata


def save_events_to_h5(data: Data, output_path: Path, metadata: dict = None):
    """Save PyG Data to HDF5 file."""
    # Convert to numpy
    events = pyg2numpy_event_convertor(data)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create events group with separate datasets (common format)
        events_group = f.create_group('events')
        events_group.create_dataset('x', data=events['x'], compression='gzip')
        events_group.create_dataset('y', data=events['y'], compression='gzip')
        events_group.create_dataset('t', data=events['t'], compression='gzip')
        events_group.create_dataset('p', data=events['p'], compression='gzip')
        
        # Save metadata
        if metadata:
            for key, value in metadata.items():
                try:
                    f.attrs[key] = value
                except:
                    pass  # Skip unsaveable metadata
        
        # Add subsampling metadata
        f.attrs['num_events'] = len(events)
        f.attrs['subsampling_method'] = 'spatiotemporal_filtering'


def process_file(input_path: Path, output_path: Path, transform, config: dict) -> dict:
    """
    Process a single file with spatiotemporal filtering.
    
    Returns:
        dict with processing stats
    """
    # Load events
    data, metadata = load_events_from_h5(input_path)
    original_count = data.pos.shape[0]
    
    # Add required metadata for transform
    data.label = [input_path.parent.name]  # split name (train/test/val)
    data.file_id = input_path.name
    
    # Apply transform
    data_subsampled = transform(data)
    final_count = data_subsampled.pos.shape[0]
    
    # Save output
    save_events_to_h5(data_subsampled, output_path, metadata)
    
    return {
        'original_count': original_count,
        'final_count': final_count,
        'ratio': final_count / original_count if original_count > 0 else 0
    }


def process_batch(batch_dir: Path, batch_id: int):
    """Process all files in a batch."""
    print(f"Processing batch {batch_id}")
    print("="*60)
    
    # Load config and batch
    config = load_config(batch_dir)
    batch = load_batch(batch_dir, batch_id)
    
    # Compute output directory
    input_dir = Path(config['input_dir'])
    output_dir_name = f"{config['output_prefix']}_tau_{config['tau']}_fs_{config['filter_size']}"
    output_dir = input_dir.parent / output_dir_name
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files in batch: {len(batch['files'])}")
    print(f"\nParameters:")
    print(f"  tau: {config['tau']} ms")
    print(f"  filter_size: {config['filter_size']}")
    print(f"  sampling_threshold: {config['sampling_threshold']}")
    
    # Get already processed files for resume
    processed_files = get_processed_files(batch_dir, batch_id)
    print(f"\nAlready processed: {len(processed_files)} files")
    
    # Create transform
    cfg_all = DictConfig({
        'dataset': {
            'image_resolution': [config['image_height'], config['image_width']],
            'name': input_dir.name,
            'dataset_path': str(input_dir)
        },
        'transform': {
            'train': {
                'spatiotemporal_filtering_subsampling': {
                    'transform': True,
                    'tau': config['tau'],
                    'filter_size': config['filter_size'],
                    'sampling_threshold': config['sampling_threshold'],
                    'normalization_length': None,
                    'mean_normalized': False
                }
            }
        }
    })
    
    cfg_dict = cfg_all['transform']['train']
    transform = SpatioTemporalFilteringSubsampling(cfg_all, cfg_dict)
    
    # Update status
    update_status(batch_dir, batch_id, 'start')
    
    # Process each file
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, file_info in enumerate(batch['files']):
        input_path = Path(file_info['input_path'])
        filename = file_info['filename']
        split = file_info['split']
        
        # Check if already processed
        if filename in processed_files:
            print(f"[{i+1}/{len(batch['files'])}] SKIP (already done): {filename}")
            skip_count += 1
            continue
        
        # Compute output path
        output_path = output_dir / split / filename
        
        print(f"[{i+1}/{len(batch['files'])}] Processing: {filename}")
        
        try:
            start_time = time.time()
            stats = process_file(input_path, output_path, transform, config)
            elapsed = time.time() - start_time
            
            print(f"  → {stats['original_count']:,} → {stats['final_count']:,} events "
                  f"({stats['ratio']*100:.1f}%) in {elapsed:.1f}s")
            
            # Save progress
            save_file_progress(batch_dir, batch_id, filename)
            success_count += 1
            
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            save_failed_file(batch_dir, batch_id, filename, str(e))
            fail_count += 1
    
    # Update final status
    if fail_count == 0:
        update_status(batch_dir, batch_id, 'complete')
    else:
        update_status(batch_dir, batch_id, 'fail')
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH COMPLETE")
    print("="*60)
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped (already done): {skip_count}")
    
    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(
        description='Worker script for HPC spatiotemporal filtering subsampling'
    )
    parser.add_argument('--batch_dir', type=str, required=True,
                        help='Path to batch directory')
    parser.add_argument('--batch_id', type=int, required=True,
                        help='Batch ID to process')
    
    args = parser.parse_args()
    
    batch_dir = Path(args.batch_dir)
    
    if not batch_dir.exists():
        print(f"Error: Batch directory does not exist: {batch_dir}")
        return 1
    
    try:
        success = process_batch(batch_dir, args.batch_id)
        return 0 if success else 1
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        update_status(batch_dir, args.batch_id, 'fail')
        return 1


if __name__ == '__main__':
    exit(main())
