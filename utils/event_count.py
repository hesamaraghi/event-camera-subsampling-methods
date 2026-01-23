#!/usr/bin/env python3
"""
Count events in h5 files and save results to JSON.

Usage:
    CLI: python -m utils.event_count /path/to/dataset [-j 32]
    API: from utils.event_count import count_events_in_dataset
"""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import h5py
from tqdm import tqdm


def _process_single_file(args: Tuple[Path, Path]) -> Optional[dict]:
    """Process a single h5 file. Designed for parallel execution."""
    h5_file, dataset_path = args
    count = count_events_in_h5(h5_file)
    if count is not None:
        return {
            "path": str(h5_file.relative_to(dataset_path)),
            "count": count,
            "town": extract_town_name(h5_file.name),
            "split": extract_split(h5_file)
        }
    return None


def count_events_in_h5(h5_path: Path) -> Optional[int]:
    """Count the number of events in an h5 file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'events' in f:
                if isinstance(f['events'], h5py.Group):
                    if 'x' in f['events']:
                        return len(f['events']['x'])
                    elif 't' in f['events']:
                        return len(f['events']['t'])
                else:
                    return len(f['events'])
            elif 'x' in f:
                return len(f['x'])
            elif 't' in f:
                return len(f['t'])
            else:
                print(f"Warning: Could not find events in {h5_path}")
                return 0
    except Exception as e:
        print(f"Error reading {h5_path}: {e}")
        return None


def extract_town_name(filename: str) -> str:
    """Extract town name from filename (e.g., Town01, Town10)."""
    match = re.search(r'Town\d+', filename)
    return match.group(0) if match else 'Unknown'


def extract_split(file_path: Path) -> str:
    """Extract split (train/val/test) from file path."""
    path_str = str(file_path)
    if '/train/' in path_str:
        return 'train'
    elif '/val/' in path_str:
        return 'val'
    elif '/test/' in path_str:
        return 'test'
    return 'unknown'


def count_events_in_dataset(
    dataset_path: str,
    output_path: Optional[str] = None,
    show_progress: bool = True,
    num_workers: Optional[int] = None
) -> dict:
    """
    Count events in all h5 files in a dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory containing h5 files
        output_path: Optional path for output JSON. If None, saves to 
                     {dataset_path}/event_counts.json
        show_progress: Whether to show progress bar
        num_workers: Number of parallel workers. Defaults to CPU count.
                     Use 1 for sequential processing.
    
    Returns:
        Dictionary containing all event counts and metadata
    """
    dataset_path = Path(dataset_path).expanduser().resolve()
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Default output path: save alongside the dataset
    if output_path is None:
        output_path = dataset_path / "event_counts.json"
    else:
        output_path = Path(output_path).expanduser().resolve()
    
    # Default to CPU count
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    
    print(f"Analyzing dataset at: {dataset_path}")
    print("Finding all h5 files...")
    
    h5_files = list(dataset_path.rglob("*.h5"))
    print(f"Found {len(h5_files)} h5 files")
    print(f"Using {num_workers} parallel workers")
    
    if len(h5_files) == 0:
        raise ValueError(f"No h5 files found in {dataset_path}")
    
    # Count events in parallel
    print("\nCounting events in each file...")
    file_data = []
    
    # Prepare args for parallel processing
    task_args = [(h5_file, dataset_path) for h5_file in h5_files]
    
    if num_workers == 1:
        # Sequential processing
        iterator = tqdm(task_args, desc="Processing files") if show_progress else task_args
        for args in iterator:
            result = _process_single_file(args)
            if result is not None:
                file_data.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_single_file, args): args for args in task_args}
            
            if show_progress:
                pbar = tqdm(total=len(futures), desc="Processing files")
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    file_data.append(result)
                if show_progress:
                    pbar.update(1)
            
            if show_progress:
                pbar.close()
    
    # Build result
    result = {
        "dataset_path": str(dataset_path),
        "total_files": len(file_data),
        "files": file_data
    }
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved event counts to: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Count events in h5 files and save results to JSON"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset directory containing h5 files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON path (default: {dataset_path}/event_counts.json)"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPUs)"
    )
    
    args = parser.parse_args()
    count_events_in_dataset(args.dataset_path, args.output, num_workers=args.jobs)


if __name__ == "__main__":
    main()
