#!/usr/bin/env python3
"""
Find optimal sampling_threshold values for spatiotemporal filtering subsampling
to achieve specific event retention ratios.

Usage:
    python find_density_thresholds.py --help
    python find_density_thresholds.py --tau 30 --filter-size 7 --target-ratios 0.38 0.18 0.09
    python find_density_thresholds.py --num-files 5 --data-dir /path/to/CARLA
"""

import argparse
import h5py
import hdf5plugin
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from omegaconf import OmegaConf, DictConfig
from datatransforms.event_transforms import SpatioTemporalFilteringSubsampling
from utils.data_utils import numpy2pyg_event_convertor, make_structured_array


# =============================================================================
# Default Configuration Values
# =============================================================================
DEFAULT_TAU_MS = 1  # Temporal decay constant in milliseconds
DEFAULT_FILTER_SIZE = 5  # Spatial Gaussian filter size
DEFAULT_NUM_FILES = 3  # Number of files to analyze
DEFAULT_TOLERANCE = 0.01  # Acceptable difference from target ratio
DEFAULT_SOURCE_THRESH = 'thresh_3'  # Source threshold folder
DEFAULT_TARGET_RATIOS = [0.38, 0.18, 0.09]  # Target retention ratios


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Find optimal sampling_threshold values for spatiotemporal filtering subsampling.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Filter parameters
    parser.add_argument(
        '--tau', type=int, default=DEFAULT_TAU_MS,
        help='Temporal decay constant in milliseconds'
    )
    parser.add_argument(
        '--filter-size', type=int, default=DEFAULT_FILTER_SIZE,
        help='Spatial Gaussian filter size (must be odd)'
    )
    
    # Data parameters
    parser.add_argument(
        '--data-dir', type=Path, default=None,
        help='Path to CARLA data directory (default: ./data/CARLA)'
    )
    parser.add_argument(
        '--source-thresh', type=str, default=DEFAULT_SOURCE_THRESH,
        help='Source threshold folder to read files from (e.g., thresh_3)'
    )
    parser.add_argument(
        '--num-files', type=int, default=DEFAULT_NUM_FILES,
        help='Number of sample files to analyze (0 = all files)'
    )
    
    # Target ratios
    parser.add_argument(
        '--target-ratios', type=float, nargs='+', default=DEFAULT_TARGET_RATIOS,
        help='Target retention ratios to find thresholds for (e.g., 0.38 0.18 0.09)'
    )
    parser.add_argument(
        '--target-names', type=str, nargs='+', default=None,
        help='Names for target ratios (e.g., thresh_5 thresh_7 thresh_9). Must match number of target-ratios.'
    )
    
    # Search parameters
    parser.add_argument(
        '--tolerance', type=float, default=DEFAULT_TOLERANCE,
        help='Acceptable difference from target ratio during binary search'
    )
    parser.add_argument(
        '--max-iterations', type=int, default=50,
        help='Maximum binary search iterations'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.filter_size % 2 == 0:
        parser.error('--filter-size must be an odd number')
    
    if args.target_names and len(args.target_names) != len(args.target_ratios):
        parser.error('--target-names must have same length as --target-ratios')
    
    # Generate default names if not provided
    if args.target_names is None:
        args.target_names = [f'ratio_{r:.0%}' for r in args.target_ratios]
    
    # Build target_ratios dict
    args.target_ratios_dict = dict(zip(args.target_names, args.target_ratios))
    
    return args


def load_carla_events(h5_file_path: Path) -> Data:
    """
    Load events from CARLA HDF5 file and convert to PyG Data object.
    
    Args:
        h5_file_path: Path to the HDF5 file
        
    Returns:
        PyG Data object with events
    """
    with h5py.File(h5_file_path, 'r') as f:
        events_data = f['events'][()]
        
    # Extract fields from structured array
    x = events_data['x']
    y = events_data['y']
    t = events_data['t'].astype(np.int64)
    p = events_data['p'].astype(bool)
    
    # Create structured array
    events_struct = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)])
    events_array = make_structured_array(x, y, t, p, dtype=events_struct)
    
    # Convert to PyG Data
    data = numpy2pyg_event_convertor(events_array)
    
    # Add metadata required by filter transforms
    data.label = [h5_file_path.parent.parent.name]  # e.g., 'CARLA'
    data.file_id = h5_file_path.name
    
    return data


def create_stf_transform(image_h: int, image_w: int, data_dir: Path,
                         tau: float, filter_size: int,
                         initial_threshold: float = 0.1) -> SpatioTemporalFilteringSubsampling:
    """
    Create a SpatioTemporalFilteringSubsampling transform.
    
    Args:
        image_h: Image height
        image_w: Image width
        data_dir: Root data directory
        tau: Temporal decay constant in milliseconds
        filter_size: Spatial Gaussian filter size
        initial_threshold: Initial sampling threshold value
        
    Returns:
        Configured SpatioTemporalFilteringSubsampling transform
    """
    cfg_all = DictConfig({
        'dataset': {
            'image_resolution': [image_h, image_w],
            'name': 'CARLA',
            'dataset_path': str(data_dir)
        },
        'transform': {
            'train': {
                'spatiotemporal_filtering_subsampling': {
                    'transform': True,
                    'tau': tau,
                    'filter_size': filter_size,
                    'sampling_threshold': initial_threshold,
                    'normalization_length': None,
                    'mean_normalized': False
                }
            }
        }
    })
    
    cfg_dict = OmegaConf.to_object(cfg_all['transform']['train'])
    return SpatioTemporalFilteringSubsampling(cfg_all, cfg_dict)


def binary_search_threshold(data: Data, target_ratio: float, transform: SpatioTemporalFilteringSubsampling,
                            tolerance: float = 0.01, max_iterations: int = 50,
                            low: float = 0.0001, high: float = 20.0) -> tuple[float, float]:
    """
    Use binary search to find sampling_threshold that achieves target ratio.
    
    Args:
        data: PyG Data object
        target_ratio: Target subsampling ratio (e.g., 0.38 for 38%)
        transform: Pre-initialized spatiotemporal filtering transform
        tolerance: Acceptable difference from target ratio
        max_iterations: Maximum search iterations
        low, high: Search range for threshold
        
    Returns:
        Tuple of (best_threshold, achieved_ratio)
    """
    best_threshold = None
    best_ratio = None
    best_diff = float('inf')
    
    print(f"  Searching for threshold to achieve {target_ratio*100:.1f}% ratio...")
    
    for iteration in range(max_iterations):
        mid = (low + high) / 2
        
        try:
            # Update threshold and apply transform
            transform.sampling_threshold = mid
            data_subsampled = transform(data.clone())
            
            original_count = data.pos.shape[0]
            final_count = data_subsampled.pos.shape[0]
            ratio = final_count / original_count
            
            diff = abs(ratio - target_ratio)
            
            if iteration % 5 == 0:
                print(f"    Iteration {iteration}: threshold={mid:.6f}, ratio={ratio*100:.2f}%")
            
            if diff < best_diff:
                best_diff = diff
                best_threshold = mid
                best_ratio = ratio
            
            if diff < tolerance:
                print(f"    ✓ Found threshold={mid:.6f} achieving ratio={ratio*100:.2f}% (target={target_ratio*100:.1f}%)")
                return mid, ratio
            
            # Adjust search range: higher threshold = MORE events retained
            # So if we have TOO MANY events, we need LOWER threshold
            if ratio > target_ratio:
                high = mid  # Too many events, reduce threshold
            else:
                low = mid   # Too few events, increase threshold
                
        except Exception as e:
            print(f"    Error at threshold={mid:.6f}: {e}")
            high = mid  # Try lower threshold
            continue
    
    print(f"    → Best found: threshold={best_threshold:.6f} achieving ratio={best_ratio*100:.2f}% (target={target_ratio*100:.1f}%, diff={best_diff*100:.2f}%)")
    return best_threshold, best_ratio


def analyze_file(file_path: Path, data_dir: Path, args: argparse.Namespace) -> dict:
    """
    Analyze a single file to find optimal thresholds.
    
    Args:
        file_path: Path to source file
        data_dir: Root CARLA data directory
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Processing: {file_path.name}")
    print(f"{'='*80}")
    
    # Load data
    print("Loading events...")
    data = load_carla_events(file_path)
    original_count = data.pos.shape[0]
    print(f"Loaded {original_count:,} events")
    
    # Get image dimensions from data
    image_h = int(data.pos[:, 1].max().item()) + 1
    image_w = int(data.pos[:, 0].max().item()) + 1
    print(f"Image dimensions: {image_w} x {image_h}")
    
    # Create transform once with initial threshold
    print(f"Initializing spatiotemporal filtering transform (tau={args.tau} ms, filter_size={args.filter_size})...")
    transform = create_stf_transform(image_h, image_w, data_dir, args.tau, args.filter_size)
    
    # Compute filter values once (this is the expensive part)
    print("Computing filter values (this may take a few minutes on first run)...")
    _ = transform(data.clone())
    print("Filter values computed and cached!")
    
    results = {
        'filename': file_path.name,
        'original_count': original_count,
        'tau': args.tau,
        'filter_size': args.filter_size,
        'thresholds': {}
    }
    
    # Find threshold for each target ratio (reusing computed filter values)
    for name, target_ratio in args.target_ratios_dict.items():
        print(f"\nFinding threshold for {name} (target: {target_ratio*100:.1f}%):")
        threshold, achieved_ratio = binary_search_threshold(
            data, target_ratio, transform,
            tolerance=args.tolerance, max_iterations=args.max_iterations
        )
        
        results['thresholds'][name] = {
            'target_ratio': target_ratio,
            'threshold': threshold,
            'achieved_ratio': achieved_ratio,
            'target_count': int(original_count * target_ratio),
            'achieved_count': int(original_count * achieved_ratio)
        }
    
    return results


def print_summary(all_results: list, args: argparse.Namespace):
    """Print summary of all results."""
    print("\n" + "="*80)
    print("SUMMARY: Optimal Sampling Thresholds for Spatiotemporal Filtering")
    print("="*80)
    print()
    
    # Aggregate results
    threshold_names = list(args.target_ratios_dict.keys())
    aggregated = {name: [] for name in threshold_names}
    
    for result in all_results:
        for name in threshold_names:
            if name in result['thresholds']:
                aggregated[name].append(result['thresholds'][name]['threshold'])
    
    # Print per-file results
    print("Per-File Results:")
    print("-"*80)
    for result in all_results:
        print(f"\n{result['filename']}:")
        print(f"  Original events: {result['original_count']:,}")
        for name in threshold_names:
            if name in result['thresholds']:
                info = result['thresholds'][name]
                print(f"  {name}: threshold={info['threshold']:.4f}, "
                      f"ratio={info['achieved_ratio']*100:.2f}%, "
                      f"events={info['achieved_count']:,}")
    
    # Print aggregated statistics
    print("\n" + "="*80)
    print("Aggregated Statistics (across all files):")
    print("="*80)
    print()
    
    for name in threshold_names:
        if aggregated[name]:
            thresholds = np.array(aggregated[name])
            mean_threshold = np.mean(thresholds)
            std_threshold = np.std(thresholds)
            min_threshold = np.min(thresholds)
            max_threshold = np.max(thresholds)
            
            # Get target ratio from first result
            target_ratio = all_results[0]['thresholds'][name]['target_ratio']
            
            print(f"{name} (target ratio: {target_ratio*100:.1f}%):")
            print(f"  Mean threshold:   {mean_threshold:.4f} (±{std_threshold:.4f})")
            print(f"  Min threshold:    {min_threshold:.4f}")
            print(f"  Max threshold:    {max_threshold:.4f}")
            print()
    
    # Print recommended values
    print("="*80)
    print("RECOMMENDED THRESHOLD VALUES:")
    print("="*80)
    print()
    tau = all_results[0]['tau']
    filter_size = all_results[0]['filter_size']
    print(f"For spatiotemporal filtering with tau={tau} ms, filter_size={filter_size}:")
    print()
    for name in threshold_names:
        if aggregated[name]:
            mean_threshold = np.mean(aggregated[name])
            target_ratio = all_results[0]['thresholds'][name]['target_ratio']
            print(f"  {name} ({target_ratio*100:.0f}% retention): "
                  f"sampling_threshold = {mean_threshold:.4f}")
    print()


def main():
    """Main execution function."""
    args = parse_args()
    
    project_root = Path(__file__).parent
    data_dir = args.data_dir if args.data_dir else project_root / 'data' / 'CARLA'
    
    if not data_dir.exists():
        print(f"Error: CARLA data directory not found at {data_dir}")
        return
    
    # Get sample files from source threshold folder
    source_dir = data_dir / args.source_thresh
    if not source_dir.exists():
        print(f"Error: Source directory not found at {source_dir}")
        return
    
    all_files = sorted(list(source_dir.glob('*.h5')))
    if args.num_files > 0:
        sample_files = all_files[:args.num_files]
    else:
        sample_files = all_files
    
    if len(sample_files) == 0:
        print(f"Error: No files found in {source_dir}")
        return
    
    print("="*80)
    print("Finding Optimal Sampling Thresholds for Spatiotemporal Filtering")
    print("="*80)
    print()
    print(f"Dataset: CARLA")
    print(f"Source folder: {args.source_thresh}")
    print(f"Files to analyze: {len(sample_files)}")
    print(f"Parameters: tau={args.tau} ms, filter_size={args.filter_size}")
    print(f"Tolerance: {args.tolerance*100:.1f}%")
    print()
    print("Target ratios:")
    for name, ratio in args.target_ratios_dict.items():
        print(f"  {name}: ~{ratio*100:.0f}% of events retained")
    
    # Analyze each file
    all_results = []
    for file_path in sample_files:
        try:
            results = analyze_file(file_path, data_dir, args)
            all_results.append(results)
        except Exception as e:
            print(f"\nError processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    if all_results:
        print_summary(all_results, args)
    else:
        print("\nNo results to summarize.")


if __name__ == "__main__":
    main()
