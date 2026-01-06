# DNA/RNA SFT Dataset Builder

This directory contains scripts for building Supervised Fine-Tuning (SFT) datasets for DNA and RNA sequences.

## Overview

The SFT dataset builder processes raw DNA/RNA sequence data with instructions and generates structural features using X3DNA, producing datasets that align with the protein and molecule SFT dataset format.

## Files

- **`build_nacid_sft_data.py`**: Main script for building DNA/RNA SFT datasets
- **`build_nacid_sft_demo.sh`**: Demo script to test the pipeline with a small sample
- **`build_nacid_encoder_data.py`**: Script for building encoder-only datasets (no instructions)
- **`seq_to_feature.py`**: Core module for converting sequences to structural features

## Input Data Format

Raw data is located in `data/nacid/raw/{dna,rna}/` with JSON files containing:

```json
{
    "task_name": "task_identifier",
    "task_type": "regression",
    "task_modality": "DNA/RNA",
    "samples": [
        {
            "sample_id": 0,
            "sequences": ["ATCG..."],
            "exchanges": [
                {"role": "USER", "message": "Question..."},
                {"role": "ASSISTANT", "message": "Answer..."}
            ],
            "seq_label": [0.05],
            "fasta_headers": ["header"]
        }
    ]
}
```

## Output Data Format

The output is saved as Parquet files in `data/sft/nacid/` with the following structure:

### Columns

- **`modality`**: 'dna' or 'rna'
- **`task_name`**: Task identifier from source data
- **`task_type`**: Type of task (e.g., 'regression')
- **`sample_id`**: Sample ID within source file
- **`sequence`**: Nucleotide sequence (up to max_seq_length)
- **`instruction`**: Formatted instruction from user-assistant exchanges
- **`seq_label`**: Original labels from source data
- **`fasta_headers`**: FASTA headers from source data
- **`source_file`**: Source JSON filename
- **`structure`**: Dictionary with structural features:
  - `seq_length`: Sequence length in nucleotides
  - `num_atoms`: Number of atoms in 3D structure
  - `node_feat`: Node features (atom types)
  - `coordinates`: 3D coordinates
  - `edge_index`: Graph edge indices
  - `edge_attr`: Edge attributes

## Usage

### Quick Demo

Run a quick demo to verify the pipeline works:

```bash
bash src/data_factory/nacid/build_nacid_sft_demo.sh
```

This processes 10 DNA samples and 10 RNA samples.

### Build DNA SFT Dataset

```bash
python3 src/data_factory/nacid/build_nacid_sft_data.py \
    --raw_data_dir data/nacid/raw \
    --output_dir data/sft/nacid \
    --modality dna \
    --max_seq_length 500 \
    --graph_radius 8.0 \
    --max_neighbors 24
```

### Build RNA SFT Dataset

```bash
python3 src/data_factory/nacid/build_nacid_sft_data.py \
    --raw_data_dir data/nacid/raw \
    --output_dir data/sft/nacid \
    --modality rna \
    --max_seq_length 500 \
    --graph_radius 8.0 \
    --max_neighbors 24
```

### Command Line Arguments

- `--raw_data_dir`: Directory containing raw JSON files (default: `data/nacid/raw`)
- `--output_dir`: Output directory for Parquet files (default: `data/sft/nacid`)
- `--modality`: 'dna' or 'rna' (required)
- `--graph_radius`: Radius for graph construction in Angstroms (default: 8.0)
- `--max_neighbors`: Maximum neighbors per node (default: 24)
- `--max_seq_length`: Maximum sequence length in nucleotides (default: 500)
- `--max_samples`: Maximum samples to process, for testing (optional)
- `--max_records_per_file`: Maximum records per Parquet file (default: 1000)
- `--fiber_exe`: Path to X3DNA fiber executable (default: 'fiber')
- `--quiet`: Suppress progress output

## Processing Pipeline

1. **Extract Samples**: Read JSON files and extract sequences with instructions
2. **Generate Structures**: For each sequence:
   - Convert T→U for RNA sequences
   - Truncate to max_seq_length if needed
   - Generate 3D structure using X3DNA fiber
   - Build graph representation (edges within graph_radius)
3. **Save to Parquet**: Save processed data with automatic file splitting

## Key Features

- **Automatic T→U Conversion**: RNA sequences in the raw data contain 'T' instead of 'U', which is automatically converted during processing
- **Sequence Truncation**: Long sequences are truncated to `max_seq_length` (default: 500 nt)
- **Instruction Formatting**: User-assistant exchanges are combined into fluent instructions
- **File Splitting**: Large datasets are automatically split into multiple files based on `max_records_per_file`
- **Error Handling**: Failed sequences are skipped with progress tracking

## Requirements

- X3DNA (specifically the `fiber` executable)
- Python packages: pandas, numpy, pyarrow, tqdm
- See `seq_to_feature.py` for structural feature generation dependencies

## Data Statistics

### DNA Dataset
- **Sources**: 30 JSON files covering various tasks:
  - Histone modifications (H3, H3K4me3, H3K27ac, etc.)
  - Chromatin accessibility
  - DNA methylation
  - Promoters (TATA, non-TATA)
  - Splice sites (donors, acceptors)
  - Enhancers
  - Plant sequences
- **Total samples**: ~5M sequences
- **Average sequence length**: 500 nt (after truncation)
- **Average atoms per structure**: ~20,500

### RNA Dataset
- **Sources**: 2 JSON files
  - Human RNA degradation
  - Mouse RNA degradation
- **Total samples**: ~21K sequences
- **Average sequence length**: 500 nt (after truncation)
- **Average atoms per structure**: ~21,300

## Alignment with Protein SFT Format

This script follows the same structure as `src/data_factory/protein_sft_v2/build_protein_sft_data.py`:

1. **Input Processing**: Extract samples from source files
2. **Structure Generation**: Generate 3D structures and graph features
3. **Instruction Handling**: Format instructions (DNA/RNA uses existing instructions, protein uses OpenAI API)
4. **Parquet Output**: Save in Parquet format with automatic file splitting
5. **Schema Alignment**: Output columns match the multimodal SFT format

## Example Output

```python
import pandas as pd

# Load DNA SFT data
df = pd.read_parquet('data/sft/nacid/dna_sft.parquet')

# Access first sample
sample = df.iloc[0]
print(f"Modality: {sample['modality']}")
print(f"Task: {sample['task_name']}")
print(f"Sequence: {sample['sequence'][:50]}...")
print(f"Instruction: {sample['instruction']}")
print(f"Structure atoms: {sample['structure']['num_atoms']}")
```

## Notes

- RNA sequences in raw data are stored with 'T' instead of 'U' - this is automatically corrected
- Very long sequences are truncated to `max_seq_length` for computational efficiency
- Structure generation may fail for some sequences - these are skipped automatically
- File sizes can be large (~2-5 MB per sample) due to structural features

## See Also

- `build_nacid_encoder_data.py`: For building encoder-only datasets without instructions
- `src/data_factory/protein_sft_v2/`: For protein SFT dataset building
- `seq_to_feature.py`: Core structural feature generation module

