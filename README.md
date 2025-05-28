# Making Every Event Count: Event Camera Subsampling Methods

[![CVPR 2025 Workshop](https://img.shields.io/badge/CVPR-2025-blue)](https://cvpr2025.thecvf.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official codebase for the CVPR 2025 workshop paper:

> **Making Every Event Count: Balancing Data Efficiency and Accuracy in Event Camera Subsampling**  
> Hesam Araghi, Jan van Gemert, Nergis Tomen  
> Delft University of Technology  
> [Paper PDF](http://arxiv.org/abs/2505.21187) | [Project Page](https://github.com/hesamaraghi/event-camera-subsampling-methods)

## 🔍 Overview

Event cameras offer high temporal resolution and power efficiency, but their high event rates can overload processing systems. We explore **six hardware-friendly event subsampling methods** and assess their impact on **event-based video classification**.

### 🔑 Key Contributions

- 📊 **Systematic evaluation** of 6 causal, hardware-friendly subsampling methods.
- 🧠 Proposal of a **causal density-based subsampling** technique to test the hypothesis that dense regions contain more information.
- 📉 Analysis of **accuracy–efficiency trade-offs** across three benchmark datasets.

## 📂 Subsampling Methods

| Type                     | Description                                  |
|--------------------------|----------------------------------------------|
| Spatial                 | Keep every *n*-th row and *m*-th column        |
| Temporal                | Keep the events in every *t*-th time interval             |
| Random                  | Keep each event with probability *p*     |
| Event Count [^1]            | Spatially subsample based on event counts within fixed-size windows   |
| Corner-based [^2]            | Keep the events in the corners using the Harris corner detector  |
| Causal Density-based    | Causally filter and keep events more in spatiotemporally dense regions   |

[^1]: A. Gruel et al., “Event Data Downscaling for Embedded Computer Vision,” VISAPP 2022.
[^2]: A. Glover et al., “luvHarris: A practical corner detector for event-cameras,” IEEE TPAMI 2021.


## 📈 Datasets

We evaluate our methods on:
- **N-Caltech101**: Object classification with camera motion.
- **DVS-Gesture**: Human gesture recognition with real motion.
- **N-Cars**: Binary classification (car vs. background) in urban driving.

## 🏗️ Setup with uv

[uv](https://github.com/astral-sh/uv) is a fast Python package manager and virtual environment tool.

### 1. Install uv

If you don't have uv, install it with:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Or follow the instructions at [uv's GitHub](https://github.com/astral-sh/uv).

### 2. Create and activate a virtual environment

```bash
uv venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Run your code

Now you can run your scripts as usual, e.g.:

```bash
python train.py --cfg_path path/to/config.yaml
```
## 📎 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{araghi2025EvSubsampling,
  title={Making Every Event Count: Balancing Data Efficiency and Accuracy in Event Camera Subsampling},
  author={Araghi, Hesam and van Gemert, Jan and Tomen, Nergis},
  booktitle={CVPR Workshop},
  year={2025}
}
```

