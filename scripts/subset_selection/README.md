# Subset Selection Scripts

The subset selection scripts use advanced machine learning techniques to identify representative samples from large datasets. It provides functionality for selecting diverse subsets of datasets using facility location maximization with embedding-based similarity. This is particularly useful for:
- Reducing dataset size while maintaining diversity
- Selecting training data that covers the full distribution
- Creating validation/test sets that represent the full dataset

## Requirements

- **Python 3.12** (required for compatibility with the rest of the codebase)
- CUDA 12.1+ for GPU support (recommended)

## Installation

Install all dependencies including PyTorch with CUDA support:

```bash
pip install -r scripts/subset_selection/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

**Note:** The CLI automatically configures multiprocessing to use the 'spawn' method for CUDA compatibility, enabling efficient multi-GPU parallel processing.

## Model Setup

The default encoder (`Snowflake/snowflake-arctic-embed-l-v2.0`) is automatically downloaded from HuggingFace on first run if not found locally. The model will be cached for subsequent runs.

### Automatic Download (Default Behavior)

On first run, the model will be automatically downloaded from HuggingFace:

```bash
# Run from scripts/subset_selection/ directory
cd odh-data-processing/scripts/subset_selection
source .venv/bin/activate  # or venv/bin/activate
python -m subset_selection \
  --input dataset.jsonl \
  --subset-sizes "10%" \
  --output-dir output/
```

The model is cached at `~/.cache/huggingface/` for future use.

### Using a Local Model (Optional)

If you have the model cached locally or want to use a custom model path:

```bash
# Run from scripts/subset_selection/ directory
cd /path/to/odh-data-processing/scripts/subset_selection
source .venv/bin/activate  # or venv/bin/activate
python -m subset_selection \
  --input dataset.jsonl \
  --subset-sizes "10%" \
  --encoder-model-path /path/to/your/local/model \
  --output-dir output/
```

### Command Line Interface (Recommended)

The easiest way to use subset selection is through the CLI (run from `scripts/subset_selection/` directory):

```bash
# Basic usage - Select 10% and 50% subsets
python -m subset_selection \
  --input path/to/dataset.jsonl \
  --subset-sizes "10%,50%" \
  --output-dir output/

# Absolute counts - Select exactly 1000 and 5000 samples
python -m subset_selection \
  --input path/to/dataset.jsonl \
  --subset-sizes "1000,5000" \
  --output-dir output/

# Small dataset (< 100k samples) - adjust epsilon and num_folds
python -m subset_selection \
  --input path/to/small_dataset.jsonl \
  --subset-sizes "50%" \
  --epsilon 0.1 \
  --num-folds 10 \
  --output-dir output/

# Multiple files combined
python -m subset_selection \
  --input file1.jsonl file2.jsonl file3.jsonl \
  --subset-sizes "25%,50%" \
  --combine-files \
  --output-dir output/

# Using a custom local model path
python -m subset_selection \
  --input dataset.jsonl \
  --subset-sizes "10%" \
  --encoder-model-path /path/to/local/model \
  --output-dir output/
```

#### CLI Options

```
Required:
  --input <file> [<file> ...]    Input file(s) to process (JSONL, JSON, CSV, Parquet)
  --subset-sizes <sizes>         Comma-separated sizes (e.g., "10%,50%" or "1000,5000")

Optional:
  --output-dir <dir>             Output directory (default: output)
  --batch-size <int>             Batch size for processing (default: 100000)
  --num-folds <int>              Number of folds/partitions (default: 50)
  --epsilon <float>              Optimization parameter (default: 160.0)
  --num-gpus <int>               Number of GPUs to use (default: auto-detect)
  --combine-files                Combine multiple input files before processing
  --encoder-type <str>           Encoder type (default: arctic)
  --encoder-model <str>          Model name (default: Snowflake/snowflake-arctic-embed-l-v2.0)
  --encoder-model-path <path>    Local path to encoder model (optional, auto-downloads if not provided)
  --template-name <str>          Template name (default: conversation)
  --seed <int>                   Random seed (default: 42)
```

#### Subset Size Formats

The `--subset-sizes` parameter accepts three formats:

1. **Percentage notation (Recommended)**: Use `"%"` for clarity
   - `"10%"` = 10% of the dataset
   - `"50%"` = 50% of the dataset
   - Example: `--subset-sizes "10%,50%,90%"`

2. **Absolute counts**: Specify exact number of samples
   - `"1000"` = exactly 1000 samples
   - `"5000"` = exactly 5000 samples
   - Example: `--subset-sizes "1000,5000"`

3. **Decimal notation (Backward compatibility)**: Float values between 0 and 1
   - `"0.1"` = 10% of the dataset
   - `"0.5"` = 50% of the dataset
   - Example: `--subset-sizes "0.1,0.5"`
   - **Note**: This format is supported for backward compatibility but percentage notation is recommended for clarity.

**Mixing formats**: You cannot mix different formats in the same command. Use either all percentages, all counts, or all decimals.

### Python API

You can also use subset selection directly in Python:

```python
from subset_selection import subset_datasets

# Select subsets from your dataset (using percentages)
subset_datasets(
    input_files=["path/to/your/dataset.jsonl"],
    subset_sizes=[0.1, 0.5],  # 10% and 50% of the dataset (as decimals)
)

# Or using absolute counts
subset_datasets(
    input_files=["path/to/your/dataset.jsonl"],
    subset_sizes=[1000, 5000],  # Exactly 1000 and 5000 samples
)
```

### Advanced Python Configuration

```python
from subset_selection import (
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
    encoder_model_path=None,  # Optional: specify local model path
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
- `encoder_model`: Model name for the encoder (default: "Snowflake/snowflake-arctic-embed-l-v2.0")
- `encoder_model_path`: Local path to encoder model (optional, will auto-download from HuggingFace if not provided)
- `instruction`: Custom instruction for embedding generation

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
└── subset_selection/
    ├── __main__.py          # Entry point for module execution
    ├── subset_selection.py  # Main subset selection logic, CLI, and encoder registry
    ├── requirements.txt     # Package dependencies
    ├── README.md            # This file
    ├── encoders/
    │   └── arctic_encoder.py  # Arctic embedding encoder
    └── utils/
        └── subset_selection_utils.py  # Utility functions
```

## Output Files

The script generates several output files:

1. **Embeddings**: Stored in HDF5 format in `{output_dir}/{dataset_name}/embeddings/`
2. **Metadata**: NPZ files containing indices and gains for each subset
3. **Subset Files**: Dataset subsets in the original file format (JSON, CSV, Parquet)

## Troubleshooting

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
  - CPU fallback is available but significantly slower (warnings will be displayed)
- **Multiple GPUs**: Automatically detects and utilizes all available GPUs
  - Uses 'spawn' multiprocessing method for CUDA compatibility
  - Override with `--num-gpus` flag if needed
- **Memory**: Each fold processes independently, so more folds = less memory per fold
- **Model Caching**: Models are automatically downloaded on first run and cached locally
  - Default cache location: `~/.cache/huggingface/`
  - Use `--encoder-model-path` to specify a custom model location
- **Performance**: 
  - Larger epsilon values = faster but potentially lower quality
  - More folds = better GPU utilization but more overhead
  - Multi-GPU processing scales linearly with the number of GPUs

## Credits and Acknowledgements

This subset selection implementation is derived from the **DataCurate4LLMs** project.

### Original Author
**Krishnateja Killamsetty**  
📫 krishnateja.killamsetty@utdallas.edu

### Original Repository
The original codebase can be found at: [https://github.com/krishnatejakk/DataCurate4LLMs](https://github.com/krishnatejakk/DataCurate4LLMs)

