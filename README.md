# Event Camera Subsampling Methods

## Setup with uv

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

