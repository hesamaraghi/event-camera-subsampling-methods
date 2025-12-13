# Event Camera Subsampling Methods

Example of how to use five event subsampling methods.

## Subsampling Methods

| Method | Description |
|--------|-------------|
| Spatial | Keep every *n*-th row and *m*-th column |
| Temporal | Keep events in every *t*-th time interval |
| Random | Keep each event with probability *p* |
| Harris Corner | Keep events at detected corner points |
| Causal Density-based (Spatiotemporal filtering) | Recursively filter and keep events in dense regions |


## Install with uv

### 1. Install uv
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Setup environment
```bash
uv sync
source .venv/bin/activate
```

## Example Notebook

Run `event_subsampling_example.ipynb` for an example of how to perform subsampling on the event data. The event data is a subset of a DSEC dataset sequence (the first 1 million events), downloaded automatically. It is in the HDF5 format.
Each section applies a subsampling method to the same event stream and saves the results in HDF5 format.

### What's Inside:
- **Section 1-2**: Setup and load event data from DSEC dataset
- **Section 3**: Spatial subsampling (keep every n-th pixel)
- **Section 4**: Temporal subsampling (keep events in time windows)
- **Section 5**: Random ratio subsampling (drop events with probability)
- **Section 6**: Spatiotemporal filtering (density-based recursive filtering with Gaussian weighting)
- **Section 7**: ToS 2D Harris corner detection (keep events at detected interest points)

### ⚠️ First Run Computation Time:
**Sections 6 (Spatiotemporal Filtering) and 7 (Harris Corner Detection)** compute filter values on first execution, which may take longer. These values are cached to disk and reused on subsequent runs, making them much faster.
