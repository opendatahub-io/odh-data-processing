# Complete Command Reference

1. Basic Usage (Default Config)
```
# Use default config with your desired subset sizes
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 10% 5%
```
2. Different Config Files

## Use Example Config (BGE Encoder)
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output \
    --config configs/replay_buffer_selection/example_config.json \
    --subset_sizes 15% 8%
```

## Use Granite Config (Arctic Encoder)
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output \
    --config configs/replay_buffer_selection/instructlab_data_config.json \
    --subset_sizes 25% 12%
```

## Use InstructLab Config (NVIDIA Encoder)
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output \
    --config configs/replay_buffer_selection/instructlab_data_config.json \
    --subset_sizes 25% 12%
```

3. Multiple Files (Process Separately)
```
# Process each file separately (creates separate subsets for each file)
python data_subset_selection.py \
    --input_files data/file1.jsonl data/file2.jsonl data/file3.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 10% 5%
```

4. Multiple Files (Combine First)
```
# Combine all files first, then create subsets from the combined dataset
python data_subset_selection.py \
    --input_files data/file1.jsonl data/file2.jsonl data/file3.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 10% 5% \
    --combine_files
```

5. Using Wildcards
```
# Process all JSONL files in data folder
python data_subset_selection.py \
    --input_files data/*.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 15% 8%

# Combine all JSONL files first
python data_subset_selection.py \
    --input_files data/*.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 15% 8% \
    --combine_files
```

6. Different Subset Size Formats
## Percentages Only
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 25% 15% 10% 5%
```

## Absolute Numbers Only
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 50 30 20 10
```

## Mixed (Percentages + Absolute)
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 20% 15% 100 50
```

7. Advanced Options
## Custom Retry Settings
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 10% 5% \
    --max_retries 5 \
    --retry_delay 60
```

## Specify Number of GPUs
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 10% 5% \
    --num_gpus 2
```

8. Different Output Directories
## Organized by Date
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output/$(date +%Y-%m-%d) \
    --config configs/default.json \
    --subset_sizes 10% 5%
```

## Organized by Config Type
```
python data_subset_selection.py \
    --input_files data/first_100_skills_train_msgs.jsonl \
    --output_dir data/output/bge_results \
    --config configs/replay_buffer_selection/example_config.json \
    --subset_sizes 10% 5%
```

# Common Use Cases
## For Small Datasets (< 1000 samples)
```
python data_subset_selection.py \
    --input_files data/small_dataset.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 20% 15% 10%
```

## For Large Datasets (> 10,000 samples)
```
python data_subset_selection.py \
    --input_files data/large_dataset.jsonl \
    --output_dir data/output \
    --config configs/replay_buffer_selection/granite3.1_config.json \
    --subset_sizes 5% 2% 1%
```

## For Multiple Related Files
```
python data_subset_selection.py \
    --input_files data/train_part1.jsonl data/train_part2.jsonl data/train_part3.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 10% 5% \
    --combine_files
```

# For Different Data Types
## Conversations
```
python data_subset_selection.py \
    --input_files data/conversations.jsonl \
    --output_dir data/output \
    --config configs/replay_buffer_selection/example_config.json \
    --subset_sizes 15% 8%
```

## Q&A Data
```
python data_subset_selection.py \
    --input_files data/qa_pairs.jsonl \
    --output_dir data/output \
    --config configs/default.json \
    --subset_sizes 20% 10%
```

