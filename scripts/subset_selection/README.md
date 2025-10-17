# Subset Selection Scripts

This package provides functionality for selecting diverse subsets of datasets using facility location maximization with embedding-based similarity.

## Overview

The subset selection scripts use advanced machine learning techniques to identify representative samples from large datasets. This is particularly useful for:
- Reducing dataset size while maintaining diversity
- Selecting training data that covers the full distribution
- Creating validation/test sets that represent the full dataset

## Installation

### 1. Install PyTorch with CUDA Support

For GPU usage (recommended), install PyTorch with CUDA support first:

```bash
# For CUDA 12.1+ (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install Other Dependencies

```bash
pip install -r scripts/subset_selection/requirements.txt
```

**Note:** The CLI automatically configures multiprocessing to use the 'spawn' method for CUDA compatibility, enabling efficient multi-GPU parallel processing.

## Model Setup

The default encoder (`Snowflake/snowflake-arctic-embed-l-v2.0`) needs to be available before running. Choose one of these options:

### Option 1: Auto-download with Testing Mode (Recommended for First Run)

Use `--testing-mode` to automatically download the model from HuggingFace:

```bash
python -m scripts.subset_selection.cli \
  --input dataset.jsonl \
  --subset-sizes "0.1" \
  --testing-mode \
  --output-dir output/
```

**Important:** Despite the name, `--testing-mode` **still uses all available GPUs** for processing. It simply allows automatic model downloading from HuggingFace and provides CPU fallback if no GPUs are detected. After the first run, the model is cached and you can omit this flag.

### Option 2: Pre-download with ilab

If you have `ilab` installed:

```bash
ilab model download --repository Snowflake/snowflake-arctic-embed-l-v2.0
```

The model will be cached at `~/.cache/instructlab/models/Snowflake/snowflake-arctic-embed-l-v2.0`

## Usage

### Command Line Interface (Recommended)

The easiest way to use subset selection is through the CLI:

```bash
# Basic usage - Select 10% and 50% subsets
python -m scripts.subset_selection.cli \
  --input path/to/dataset.jsonl \
  --subset-sizes "0.1,0.5" \
  --output-dir output/

# Absolute counts - Select exactly 1000 and 5000 samples
python -m scripts.subset_selection.cli \
  --input path/to/dataset.jsonl \
  --subset-sizes "1000,5000" \
  --output-dir output/

# Small dataset (< 100k samples) - adjust epsilon and num_folds
python -m scripts.subset_selection.cli \
  --input path/to/small_dataset.jsonl \
  --subset-sizes "0.5" \
  --epsilon 0.1 \
  --num-folds 10 \
  --output-dir output/

# Multiple files combined
python -m scripts.subset_selection.cli \
  --input file1.jsonl file2.jsonl file3.jsonl \
  --subset-sizes "0.25,0.5" \
  --combine-files \
  --output-dir output/

# Testing mode (allows model auto-download, still uses GPUs if available)
python -m scripts.subset_selection.cli \
  --input dataset.jsonl \
  --subset-sizes "0.1" \
  --testing-mode \
  --output-dir output/
```

#### CLI Options

```
Required:
  --input <file> [<file> ...]    Input file(s) to process (JSONL, JSON, CSV, Parquet)
  --subset-sizes <sizes>         Comma-separated sizes (e.g., "0.1,0.5" or "1000,5000")

Optional:
  --output-dir <dir>             Output directory (default: output)
  --batch-size <int>             Batch size for processing (default: 100000)
  --num-folds <int>              Number of folds/partitions (default: 50)
  --epsilon <float>              Optimization parameter (default: 160.0)
  --num-gpus <int>               Number of GPUs to use (default: auto-detect)
  --combine-files                Combine multiple input files before processing
  --testing-mode                 Enable model auto-download and CPU fallback
  --encoder-type <str>           Encoder type (default: arctic)
  --encoder-model <str>          Model name (default: Snowflake/snowflake-arctic-embed-l-v2.0)
  --template-name <str>          Template name (default: conversation)
  --seed <int>                   Random seed (default: 42)
```

### Python API

You can also use subset selection directly in Python:

```python
from scripts import subset_datasets

# Select subsets from your dataset
subset_datasets(
    input_files=["path/to/your/dataset.jsonl"],
    subset_sizes=[0.1, 0.5],  # 10% and 50% of the dataset
)
```

### Advanced Python Configuration

```python
from scripts import (
    subset_datasets,
    BasicConfig,
    EncoderConfig,
    TemplateConfig,
    SystemConfig
)

# Configure subset selection
subset_datasets(
    input_files=["dataset1.jsonl", "dataset2.jsonl"],
    subset_sizes=[1000, 5000],  # Select 1000 and 5000 samples
    output_dir="output",
    batch_size=100000,
    num_folds=50,
    combine_files=False,
    epsilon=160.0,
    encoder_type="arctic",
    encoder_model="Snowflake/snowflake-arctic-embed-l-v2.0",
    template_name="conversation",
)
```

## Configuration

### BasicConfig Parameters

- **`output_dir`**: Directory for output files (default: `"output"`)
- **`batch_size`**: Batch size for processing (default: `100000`)
- **`num_folds`**: Number of folds/partitions for subset selection (default: `50`)
  - The dataset is divided into folds for parallel processing across GPUs
  - **Recommendations based on dataset size:**
    - < 1,000 samples: Use `5-10` folds
    - 1,000-10,000 samples: Use `10-20` folds
    - 10,000-100,000 samples: Use `20-50` folds
    - \> 100,000 samples: Use `50-100` folds (default: 50)
  - More folds = better parallelization but higher memory usage per fold
  - Use fewer folds for small datasets to ensure each fold has enough samples
- **`combine_files`**: Whether to combine multiple input files (default: `False`)
- **`epsilon`**: Epsilon parameter for the LazierThanLazyGreedy optimizer (default: `160.0`)
  - Controls the trade-off between optimization quality and speed
  - **Recommendations based on dataset size:**
    - < 1,000 samples: Use `0.01-0.1`
    - 1,000-10,000 samples: Use `0.1-1.0`
    - 10,000-100,000 samples: Use `1.0-10.0`
    - \> 100,000 samples: Use `160.0` (default)

### EncoderConfig Parameters

- `encoder_type`: Type of encoder to use (default: "arctic")
- `encoder_model`: Model name for the encoder
- `instruction`: Custom instruction for embedding generation
- `testing_mode`: Enable model auto-download from HuggingFace and CPU fallback (default: False)

### TemplateConfig Parameters

- `template_name`: Name of the template to use (default: "conversation")
- `templates`: Custom templates for text formatting

### SystemConfig Parameters

- `num_gpus`: Number of GPUs to use (auto-detected by default)
- `seed`: Random seed for reproducibility (default: 42)
- `max_retries`: Maximum number of retries on failure (default: 3)
- `retry_delay`: Delay between retries in seconds (default: 30)

## Package Structure

```
scripts/
├── __init__.py              # Top-level package initialization
└── subset_selection/
    ├── __init__.py          # Subset selection package initialization
    ├── subset_selection.py  # Main subset selection logic
    ├── cli.py              # Command-line interface
    ├── requirements.txt    # Package dependencies
    ├── README.md          # This file
    ├── encoders/
    │   ├── __init__.py     # Encoder registry
    │   └── arctic_encoder.py  # Arctic embedding encoder
    └── utils/
        ├── __init__.py     # Utils initialization
        └── subset_selection_utils.py  # Utility functions
```

## Supported Encoders

Currently supported encoders:
- `arctic`: Snowflake Arctic Embed models

To see all supported encoders:

```python
from scripts import get_supported_encoders
print(get_supported_encoders())
```

## Output Files

The script generates several output files:

1. **Embeddings**: Stored in HDF5 format in `{output_dir}/{dataset_name}/embeddings/`
2. **Metadata**: NPZ files containing indices and gains for each subset
3. **Subset Files**: Dataset subsets in the original file format (JSON, CSV, Parquet)


## Quick Start Example

Using your data file:

```bash
# Navigate to project root
cd /Users/roburishabh/Github/odh-data-processing

# Run subset selection - Select 10% and 50% subsets
python -m scripts.subset_selection.cli \
  --input scripts/subset_selection/data/combined_cut_50x.jsonl \
  --subset-sizes "0.1,0.5" \
  --output-dir scripts/subset_selection/data/output \
  --epsilon 0.1 \
  --num-folds 10

# Check results
ls scripts/subset_selection/data/output/
```

## Troubleshooting

### CUDA Multiprocessing Errors

The CLI automatically handles CUDA multiprocessing compatibility by setting the start method to 'spawn' (required on Linux). If you're using the Python API directly and encounter errors like:
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

Add this at the start of your script:
```python
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```

### Model Not Found Error

If you see `Model not found in available models: Snowflake/snowflake-arctic-embed-l-v2.0`, you have two options:

1. **Use `--testing-mode`** to auto-download from HuggingFace (still uses GPUs):
   ```bash
   python -m scripts.subset_selection.cli --input data.jsonl --subset-sizes "0.1" --testing-mode --output-dir output/
   ```

2. **Pre-download with ilab**:
   ```bash
   ilab model download --repository Snowflake/snowflake-arctic-embed-l-v2.0
   ```

### Memory Issues

If you run out of GPU memory:
- Reduce `--num-folds` to process larger chunks per GPU
- Reduce `--num-gpus` to use fewer GPUs
- For small datasets (<10k samples), use fewer folds (5-10)
- The default batch size is optimized for A100 GPUs; adjust if needed

### GPU Not Detected

Verify CUDA is properly installed and accessible:
```bash
# Check GPU availability
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Notes

- **Dataset Size**: Subset selection is optimized for datasets >100k samples
  - For smaller datasets, adjust `--epsilon` and `--num-folds` accordingly
- **GPU Requirement**: GPU acceleration is strongly recommended for production use
  - The code automatically uses all available GPUs with parallel processing
  - CPU fallback available with `--testing-mode` (much slower)
- **Multiple GPUs**: Automatically detects and utilizes all available GPUs
  - Uses 'spawn' multiprocessing method for CUDA compatibility
  - Override with `--num-gpus` flag if needed
- **Memory**: Each fold processes independently, so more folds = less memory per fold
- **Performance**: 
  - Larger epsilon values = faster but potentially lower quality
  - More folds = better GPU utilization but more overhead
  - Multi-GPU processing scales linearly with the number of GPUs

