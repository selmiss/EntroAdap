# Protein Encoder Dataset Builder

This directory contains tools to build protein encoder pretraining datasets from UniProt JSON files with PDB structural data.

## Overview

The pipeline:
1. Reads UniProt JSON files (containing PDB cross-references)
2. Extracts all unique PDB IDs (~67k structures from your data)
3. Downloads missing PDB structures automatically
4. Processes each structure to extract:
   - 3D atomic coordinates
   - Atom features (element, atom type, residue, chain, etc.)
   - Spatial graph (edges within radius threshold)
5. Saves to Parquet format for efficient loading

## Quick Start

### Test with 100 structures:
```bash
cd /home/UWO/zjing29/proj/EntroAdap
bash src/data_factory/protein/build_encoder_dataset.sh --max_structures 100
```

### Build full dataset (all ~67k structures):
```bash
bash src/data_factory/protein/build_encoder_dataset.sh
```

This will:
- Read all 5 UniProt JSON files from `data/uniprot/full/`
- Extract 67,731 unique PDB IDs
- Download missing structures to `data/pdb_structures/`
- Process and save to `data/encoder/protein_pretrain.parquet`
- Use checkpointing (resume if interrupted)

## Command Line Options

```bash
python src/data_factory/protein/build_protein_encoder_data.py [OPTIONS]
```

### Key Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--uniprot_json_dir` | `data/uniprot/full` | Directory with UniProt JSON files |
| `--structure_dir` | `data/pdb_structures` | Directory for PDB CIF files |
| `--output_file` | `data/encoder/protein_pretrain.parquet` | Output parquet file |
| `--all_atoms` | False | Extract all atoms (default: C-alpha only) |
| `--graph_radius` | 8.0 | Radius for graph edges (Angstroms) |
| `--max_neighbors` | 24 | Max neighbors per node |
| `--num_workers` | auto | Parallel workers (auto = CPU-1) |
| `--batch_size` | 50 | Batch size for multiprocessing |
| `--checkpoint_interval` | 1000 | Save checkpoint every N structures |
| `--no_resume` | False | Start from scratch (ignore checkpoint) |
| `--no_download` | False | Skip downloading missing structures |
| `--max_structures` | None | Limit for testing (e.g., 100) |

### Examples:

**All atoms (heavy) instead of C-alpha only:**
```bash
python src/data_factory/protein/build_protein_encoder_data.py \
    --all_atoms \
    --graph_radius 6.0 \
    --output_file data/encoder/protein_allatom.parquet
```

**High parallelism:**
```bash
python src/data_factory/protein/build_protein_encoder_data.py \
    --num_workers 32 \
    --batch_size 100
```

**Test run with 50 structures:**
```bash
python src/data_factory/protein/build_protein_encoder_data.py \
    --max_structures 50 \
    --output_file data/encoder/protein_test.parquet
```

## Output Format

Parquet file with columns:

| Column | Type | Description |
|--------|------|-------------|
| `modality` | str | Always "protein" |
| `pdb_id` | str | PDB identifier (e.g., "6H86") |
| `uniprot_id` | str | UniProt accession (e.g., "A1L190") |
| `method` | str | Experimental method (e.g., "X-ray", "EM", "NMR") |
| `resolution` | str | Resolution (e.g., "1.90 A") |
| `chains` | str | Chain coverage (e.g., "A/B=1-88") |
| `num_atoms` | int | Number of atoms in structure |
| `node_feat` | array | Node features [N, 7] - int64 |
| `coordinates` | array | 3D coordinates [N, 3] - float32 |
| `edge_index` | array | Edge connectivity [2, E] - int64 |
| `edge_attr` | array | Edge distances [E, 1] - float32 |

### Node Features (7 dimensions):
1. **atomic_number** (0-118): Element atomic number
2. **atom_name_idx**: Index in atom name vocabulary (CA, N, C, etc.)
3. **residue_name_idx**: Index in residue vocabulary (ALA, GLY, etc.)
4. **chain_idx**: Chain identifier index
5. **residue_id**: Residue sequence number
6. **is_backbone**: Binary flag (1 if N/CA/C/O)
7. **is_ca**: Binary flag (1 if C-alpha)

## Data Statistics (from your UniProt JSON files)

- **Total proteins**: 8,181
- **Total PDB records**: 107,634
- **Unique PDB structures**: 67,731
- **PDB IDs per protein**:
  - Mean: 13.2
  - Median: 4
  - Range: 1-1,245

## Performance

On your dataset (~67k structures):
- **Estimated time**: 3-6 hours with 16 workers (depends on download speed)
- **Estimated size**: ~15-30 GB (C-alpha only), ~100-200 GB (all atoms)
- **Checkpointing**: Auto-saves every 1000 structures (resume if interrupted)

## Notes

1. **Multiple conformers**: One UniProt protein → multiple PDB IDs. All unique PDB structures are processed (no deduplication by protein).

2. **Filtering**: By default, only protein atoms are included (waters, ions, solvents excluded). Complexes with DNA/RNA/ligands will include all protein chains but exclude non-protein entities.

3. **C-alpha vs all-atom**:
   - **C-alpha** (default): ~100-200 atoms/structure, faster, good for coarse-grained models
   - **All-atom**: ~1000-3000 atoms/structure, slower, full detail

4. **Graph construction**: Uses k-NN + radius filtering for O(N·k) complexity. Adjust `--graph_radius` and `--max_neighbors` based on your model.

5. **Resume capability**: If interrupted, rerun the same command to continue from last checkpoint.

## Troubleshooting

**Out of memory during multiprocessing:**
```bash
--num_workers 4 --batch_size 20
```

**Download failures:**
- Some PDB IDs may be obsolete/unavailable. Script skips failures automatically.
- Check `data/pdb_structures/` for downloaded files.

**Checkpoint corruption:**
```bash
--no_resume  # Start fresh
```

## Integration with Training

Load in your training script:
```python
import pandas as pd
import numpy as np

df = pd.read_parquet('data/encoder/protein_pretrain.parquet')

for idx, row in df.iterrows():
    node_feat = np.array(row['node_feat'])       # [N, 7]
    coordinates = np.array(row['coordinates'])    # [N, 3]
    edge_index = np.array(row['edge_index'])     # [2, E]
    edge_attr = np.array(row['edge_attr'])       # [E, 1]
    # ... your training code
```

