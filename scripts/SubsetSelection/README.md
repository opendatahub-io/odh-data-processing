# SubsetSelection

SubsetSelection is a sophisticated data curation toolkit designed for Large Language Models (LLMs). It enables efficient training by reducing dataset size without significant loss of information, leveraging advanced embedding techniques and submodular optimization.

## **Overview**

SubsetSelection helps you select representative subsets from large datasets that can be used for:

- **Configs** for continual learning
- **Representative subsets** for efficient fine-tuning
- **Data redundancy analysis**
- **Training time optimization** by reducing dataset size
- **Diversity analysis** of datasets

The core goal is to **enable efficient training by reducing dataset size without significant loss of information**.

## **Features**

- **Multiple Encoders**: BGE, OpenAI, NVIDIA NV-Embed, Arctic, Qwen2, and more
- **GPU Acceleration**: Parallel processing with automatic GPU detection
- **Smart Submodular Optimization**: Uses Facility Location with LazierThanLazyGreedy optimizer
- **Flexible Subset Sizes**: Support for both percentages and absolute counts
- **Multiple Data Formats**: JSONL, JSON, CSV, Parquet
- **Customizable Templates**: Built-in templates for conversations, Q&A, and documents
- **Fault Tolerance**: Automatic retry mechanisms and error handling
- **Batch Processing**: Handle multiple files separately or combined

## **Quick Start**

### **Installation**

```bash
# Clone the repository
git clone https://github.com/opendatahub-io/odh-data-processing.git
cd scripts/SubsetSelection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Basic Usage**

```bash
# Simple subset selection
python data_subset_selection.py \
    --input_files data/your_dataset.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 10% 5%
```

## **Command Reference**

### **Required Arguments**

- `--input_files`: One or more input files to process (space-separated)
- `--output_dir`: Directory where output files will be saved
- `--config`: Path to configuration JSON file

### **Common Options**

- `--subset_sizes`: Subset sizes (e.g., `10% 5%` or `1000 500`)
- `--combine_files`: Combine all input files before selection
- `--num_gpus`: Number of GPUs to use (default: auto-detect)

## **Configuration**

### **Available Config Files**

| Config File | Encoder | Use Case | Subset Sizes |
|-------------|---------|----------|--------------|
| `configs/default.json` | BGE | General purpose | 30%, 20% |
| `configs/conversation.json` | Arctic | Large datasets | 1%, 5%, 10%, 25%, 50% |
| `configs/qa.json` | NVIDIA | InstructLab data | 20% |

### **Configuration Parameters**

```json
{
  "instruction": "Generate embeddings that capture core meaning...",
  "query_description": "Conversation",
  "templates": {
    "default": "{{ text }}",
    "conversation": "{% for conv in conversations %}{{ conv.from }}: {{ conv.value }}\n{% endfor %}",
    "qa": "Question: {{ question }}\nAnswer: {{ answer }}"
  },
  "batch_size": 100000,
  "num_folds": 5,
  "subset_sizes": ["10%", "5%"],
  "seed": 42,
  "template_name": "conversation",
  "combine_files": false,
  "encoder_type": "bge",
  "encoder_model": "BAAI/bge-m3",
  "epsilon": 160
}
```

## **Usage Examples**

### **1. Basic Subset Selection**
```bash
python data_subset_selection.py \
    --input_files data/dataset.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 10% 5%
```

### **2. Multiple Files (Process Separately)**
```bash
python data_subset_selection.py \
    --input_files data/file1.jsonl data/file2.jsonl data/file3.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 15% 8%
```

### **3. Multiple Files (Combine First)**
```bash
python data_subset_selection.py \
    --input_files data/file1.jsonl data/file2.jsonl data/file3.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 15% 8% \
    --combine_files
```

### **4. Using Wildcards**
```bash
# Process all JSONL files
python data_subset_selection.py \
    --input_files data/*.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 20% 10%

# Combine all files first
python data_subset_selection.py \
    --input_files data/*.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 20% 10% \
    --combine_files
```

### **5. Different Encoders**
```bash
# Use Arctic encoder
python data_subset_selection.py \
    --input_files data/dataset.jsonl \
    --output_dir data/output \
    --config configs/replay_buffer_selection/granite3.1_config.json \
    --subset_sizes 10% 5%

# Use NVIDIA encoder
python data_subset_selection.py \
    --input_files data/dataset.jsonl \
    --output_dir data/output \
    --config configs/replay_buffer_selection/instructlab_data_config.json \
    --subset_sizes 15% 8%
```

### **6. Mixed Subset Sizes**
```bash
# Combine percentages and absolute counts
python data_subset_selection.py \
    --input_files data/dataset.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 20% 15% 1000 500
```

### **7. Advanced Options**
```bash
python data_subset_selection.py \
    --input_files data/dataset.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 10% 5% \
    --num_gpus 2 \
    --max_retries 5 \
    --retry_delay 60
```

## **Repository Structure**
```
SubsetSelection/
├── data_subset_selection.py # Main script
├── configs/ # Configuration files
│ ├── default.json # Default configuration
│ ├── conversation.json # Arctic encoder config
│ └── qa.json # NVIDIA encoder config
├── src/ # Source code
│ ├── encoders/ # Embedding encoders
│ │ ├── base.py # Base encoder class
│ │ ├── bge.py # BGE encoder
│ │ ├── openai.py # OpenAI encoder
│ │ ├── nvembed.py # NVIDIA encoder
│ │ ├── arctic.py # Arctic encoder
│ │ ├── qwen2.py # Qwen2 encoder
│ │ ├── sentence.py # Sentence transformer
│ │ └── sfr_mistral.py # SFR Mistral encoder
│ └── utils/ # Utility functions
│   └── compute_pairwise_similarity.py # Similarity computation
├── requirements.txt # Dependencies
└── README.md # This file
```


## **Advanced Configuration**

### **Parameter Tuning Guidelines**

| Parameter | Small Datasets (<1K) | Medium Datasets (1K-10K) | Large Datasets (>10K) |
|-----------|----------------------|---------------------------|------------------------|
| `num_folds` | 1 | 5-10 | 10-25 |
| `batch_size` | 10,000 | 50,000 | 100,000 |
| `epsilon` | 0.01-1.0 | 10-50 | 160 |
| `subset_sizes` | 20%, 15%, 10% | 15%, 10%, 5% | 10%, 5%, 2% |

### **Encoder Selection Guide**

| Encoder | Best For | Model Size | Speed |
|---------|----------|------------|-------|
| BGE | General purpose, multilingual | Medium | Fast |
| Arctic | Large datasets, high quality | Large | Medium |
| NVIDIA NV-Embed | Retrieval tasks | Large | Medium |
| OpenAI | API-based, no local GPU needed | N/A | Slow |
| Qwen2 | Instruction-following data | Large | Medium |

### **Template Customization**

```json
{
  "templates": {
    "custom": "Document: {{ title }}\nContent: {{ content }}",
    "chat": "User: {{ user_message }}\nAssistant: {{ assistant_message }}",
    "code": "Language: {{ language }}\nCode: {{ code }}\nDescription: {{ description }}"
  }
}
```

## **Output Structure**

After running subset selection, you'll get:
```
output/
├── dataset_name/
│ ├── embeddings/
│ │ └── embeddings.h5 # Generated embeddings
│ ├── dataset_name_percent_10.0_subset.jsonl # 10% subset
│ ├── dataset_name_percent_5.0_subset.jsonl # 5% subset
│ └── dataset_name_samples_1000_subset.jsonl # 1000 samples subset
├── dataset_name_fl_5_partitions_percent_10.0_metadata.npz # Metadata
└── dataset_name_fl_5_partitions_percent_5.0_metadata.npz # Metadata
```


## **Attribution**

This SubsetSelection toolkit is based on original work by [Krishnateja Killamsetty](https://github.com/krishnatejakk). The code has been adapted and integrated into the ODH data processing pipeline.

**Original Repository**: [https://github.com/krishnatejakk/DataCurate4LLMs.git](https://github.com/krishnatejakk/DataCurate4LLMs.git)


---

**Happy subset selecting!** 
