# Protein SFT Dataset Builder (v2)

This directory contains scripts for building protein supervised fine-tuning (SFT) datasets from UniProt JSON files with structural features.

## Overview

The `build_protein_sft_data.py` script processes UniProt protein data to create a comprehensive SFT dataset that includes:
- **Structural features**: 3D coordinates, atom information, and graph edges from PDB structures
- **Text instructions**: Fluent descriptions generated from UniProt comments using OpenAI API
- **Metadata**: UniProt ID, PDB ID, sequence, resolution, method, etc.

## Key Features

- **Filters proteins with single PDB**: Only processes proteins that have exactly one PDB structure
- **Structural feature extraction**: Uses `pdbid_to_features` to extract C-alpha atoms, coordinates, and graph edges
- **Instruction construction**: Converts structured UniProt comments into fluent text using OpenAI GPT models
- **Parquet output with splitting**: Saves data in efficient Parquet format with automatic file splitting
- **Two API modes**: 
  - Sequential mode: One-by-one API calls (for testing)
  - Batch mode: Batch API requests (for production)
- **Automatic PDB download**: Downloads missing PDB structures with rate limiting

## Usage

### Test with demo data (5 samples)
```bash
./test_demo.sh
```

### Run on full dataset
```bash
./run_full_production.sh
```

### Custom run
```bash
python3 build_protein_sft_data.py \
    --uniprot_json_dir data/uniprot/full \
    --structure_dir data/pdb_structures \
    --output_dir data/sft/protein \
    --max_samples 100 \
    --sequential_api \
    --model gpt-5-mini \
    --max_records_per_file 1000
```

## Command-line Options

- `--uniprot_json_dir`: Directory containing UniProt JSON files (default: `data/uniprot/full`)
- `--structure_dir`: Directory for PDB CIF files (default: `data/pdb_structures`)
- `--output_dir`: Output directory for Parquet files (default: `data/sft/protein`)
- `--max_records_per_file`: Maximum records per Parquet file (default: 1000)
- `--all_atoms`: Extract all atoms instead of C-alpha only
- `--graph_radius`: Radius for graph construction in Angstroms (default: 8.0)
- `--max_neighbors`: Maximum neighbors per node (default: 24)
- `--no_download`: Skip downloading missing PDB structures
- `--download_delay`: Delay between PDB requests in seconds (default: 0.1)
- `--sequential_api`: Use sequential API instead of batch API
- `--model`: OpenAI model to use (default: `gpt-5-mini`)
- `--max_samples`: Maximum samples to process (for testing)
- `--quiet`: Suppress progress output

## Output Format

The output is Parquet files in the specified directory. If the dataset is large, it will be automatically split into multiple files (e.g., `protein_sft_part000.parquet`, `protein_sft_part001.parquet`, etc.).

Each record contains:

```python
{
  "modality": "protein",
  "uniprot_id": "A1L190",
  "pdb_id": "6h86",
  "method": "X-ray",
  "resolution": "1.90 A",
  "chains": "A/B=1-88",
  "sequence": "MDDADPEERNYDNMLK...",
  "structure": {
    "num_atoms": 146,
    "node_feat": [[6,1,15,0,10,1,1], ...],    # N x 7
    "coordinates": [[x,y,z], ...],             # N x 3
    "edge_index": [[sources], [targets]],      # 2 x E
    "edge_attr": [[distance], ...]             # E x 1
  },
  "instruction": "This protein is a core component of..."
}
```

Node features (7D vector):
  [atom_type, residue_index, residue_type, chain_index, 
   residue_number, model_number, hetatm_flag]

## Loading Data

```python
import pandas as pd

# Load single file
df = pd.read_parquet('data/sft/protein/protein_sft.parquet')

# Load all files from directory
from pathlib import Path
parquet_files = sorted(Path('data/sft/protein').glob('*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

# Access structure data
record = df.iloc[0]
print(f"UniProt ID: {record['uniprot_id']}")
print(f"Number of atoms: {record['structure']['num_atoms']}")
print(f"Coordinates shape: {len(record['structure']['coordinates'])}")
```

## Data Processing Pipeline

1. **Extract protein records**: Scan UniProt JSON files and filter proteins with exactly one PDB ID
2. **Download structures**: Automatically download missing PDB CIF files with rate limiting
3. **Extract structural features**: Process PDB structures to generate 3D coordinates and graph data
4. **Generate instructions**: Use OpenAI API to convert structured comments into fluent text
5. **Save dataset**: Combine all data and save to Parquet format with automatic splitting

## Validation

Use `inspect_output.py` to validate and inspect the output:

```bash
# Inspect directory with Parquet files
python3 inspect_output.py data/sft/protein

# Show detailed instructions
python3 inspect_output.py data/sft/protein --show-instructions --max-display 3
```

## Dependencies

- `pdbid_to_feature.py`: PDB structure feature extraction
- `map_fetch_pdb3d.py`: PDB structure downloading
- `utils.gpt_helper.openai_api`: OpenAI API wrappers
- pandas: For Parquet file I/O
- OpenAI API key must be configured

## Notes

- The script filters for proteins with **exactly one PDB ID** to avoid ambiguity
- Structural features use C-alpha atoms by default (use `--all_atoms` for full atom extraction)
- Batch API is recommended for large-scale production runs (up to 24h completion window)
- Sequential API is useful for testing and debugging with small samples
- PDB download respects rate limits (default: 0.1s delay = ~10 req/sec)
- Files are automatically split to keep manageable sizes (default: 1000 records per file)

## Examples

### Demo test (5 samples, sequential API)
```bash
python3 build_protein_sft_data.py \
    --max_samples 5 \
    --sequential_api \
    --model gpt-5-mini \
    --output_dir data/sft/protein_demo
```

### Production run (batch API, full dataset)
```bash
python3 build_protein_sft_data.py \
    --uniprot_json_dir data/uniprot/full \
    --output_dir data/sft/protein \
    --model gpt-5-mini \
    --max_records_per_file 1000
```

### Custom parameters
```bash
python3 build_protein_sft_data.py \
    --graph_radius 10.0 \
    --max_neighbors 32 \
    --all_atoms \
    --max_samples 1000 \
    --output_dir data/sft/protein_custom
```

Dataset ready for downstream training pipelines! 🎉
