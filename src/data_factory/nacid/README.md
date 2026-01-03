# Nucleic Acid (DNA/RNA) Encoder Dataset Generation

This directory contains scripts for generating 3D structural data from DNA and RNA sequences for encoder pretraining.

## Overview

The pipeline converts nucleic acid sequences (DNA/RNA) into 3D structures using X3DNA's `fiber` tool, then extracts atomic features and builds spatial graphs for training.

## Requirements

- **X3DNA**: Required for generating 3D structures from sequences
  - Download from: http://x3dna.org/
  - Ensure `fiber` command is in PATH
  - Or set `X3DNA_FIBER` environment variable
- **Python packages**: numpy, pandas, scipy, pathlib, tqdm

## Input Data Format

Input files are plain text files with one sequence per line:

```
ATTCAGATTGCCTCTCATTGTCTCACCC
CCCAGTCCCACACCGCAGCAGCGCCTCAGCACCGCGACTTGCCGGAGC
GCATCACAAGGCTAGTCTTTATCTTTCCGGCATCCGTACGAGTTGAAA
...
```

**Expected locations:**
- DNA sequences: `data/nacid/seq_only/dna/*.txt`
- RNA sequences: `data/nacid/seq_only/rna/*.txt`

## Output Data Format

Output is stored in Apache Parquet format with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `modality` | string | "dna" or "rna" |
| `seq_id` | string | Unique sequence identifier |
| `source_file` | string | Source filename |
| `sequence` | string | Original nucleotide sequence |
| `seq_length` | int | Length of sequence |
| `num_atoms` | int | Number of atoms in structure |
| `node_feat` | ndarray[int] | Node features, shape: `[N, 7]` |
| `coordinates` | ndarray[float] | 3D coordinates (Å), shape: `[N, 3]` |
| `edge_index` | ndarray[int] | Edge connectivity (COO format), shape: `[2, E]` |
| `edge_attr` | ndarray[float] | Edge distances (Å), shape: `[E, 1]` |

### Node Features (`node_feat`)

Shape: `[N, 7]` where N is the number of atoms.

| Index | Feature | Vocabulary Size | Description |
|-------|---------|----------------|-------------|
| 0 | `atomic_number` | 119 | Atomic number (0-118) |
| 1 | `atom_name` | 30 | Atom name (P, C1', N1, etc.) |
| 2 | `nucleotide` | 10 | Nucleotide type (A, C, G, T, U) |
| 3 | `chain` | 27 | Chain identifier |
| 4 | `residue_id` | continuous | Residue sequence number |
| 5 | `is_backbone` | 2 | Is backbone atom (0/1) |
| 6 | `is_phosphate` | 2 | Is phosphate atom (0/1) |

**Note:** Feature order matches protein features (first feature is atomic number) for consistency.

## Usage

### Quick Start

Run the pipeline on full datasets:

```bash
cd /home/UWO/zjing29/proj/EntroAdap
source local_env.sh  # Initialize environment
bash src/data_factory/nacid/build_encoder_dataset.sh
```

### Test Mode

Run on a small subset (100 sequences per file) for testing:

```bash
bash src/data_factory/nacid/build_encoder_dataset.sh --test
```

### Individual Dataset

Process DNA or RNA separately:

```bash
# DNA only
python src/data_factory/nacid/build_nacid_encoder_data.py \
    --input_dir data/nacid/seq_only/dna \
    --output_file data/encoder/nacid/dna_encoder.parquet \
    --seq_type dna \
    --graph_radius 8.0 \
    --max_neighbors 24 \
    --num_workers 8

# RNA only
python src/data_factory/nacid/build_nacid_encoder_data.py \
    --input_dir data/nacid/seq_only/rna \
    --output_file data/encoder/nacid/rna_encoder.parquet \
    --seq_type rna \
    --graph_radius 8.0 \
    --max_neighbors 24 \
    --num_workers 8
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | (required) | Directory containing sequence text files |
| `--output_file` | (required) | Output parquet file path |
| `--seq_type` | (required) | "dna" or "rna" |
| `--graph_radius` | 8.0 | Radius for graph construction (Å) |
| `--max_neighbors` | 24 | Maximum neighbors per node |
| `--num_workers` | auto | Number of parallel workers |
| `--batch_size` | 100 | Batch size for multiprocessing |
| `--checkpoint_interval` | 1000 | Save checkpoint every N sequences |
| `--no_resume` | false | Start from scratch (ignore checkpoint) |
| `--max_sequences` | none | Limit number of sequences (for testing) |
| `--fiber_exe` | fiber | Path to X3DNA fiber executable |

## Pipeline Steps

1. **Sequence Loading**: Read sequences from text files
2. **3D Structure Generation**: Use X3DNA `fiber` to generate B-DNA (DNA) or A-RNA (RNA) structures
3. **Feature Extraction**: Parse PDB files and extract atomic features
4. **Graph Construction**: Build radius graph with spatial edges
5. **Data Serialization**: Save to parquet format with checkpointing

## Testing

Test the feature extraction module:

```bash
cd /home/UWO/zjing29/proj/EntroAdap
source local_env.sh
python src/data_factory/nacid/seq_to_feature.py
```

Expected output:
```
Testing DNA sequence processing...
✓ Successfully processed DNA sequence
  Modality: dna
  Sequence length: 28
  Number of atoms: ~600
  ...

Testing RNA sequence processing...
✓ Successfully processed RNA sequence
  Modality: rna
  Sequence length: 29
  Number of atoms: ~650
  ...
```

## File Structure

```
src/data_factory/nacid/
├── README.md                      # This file
├── seq_to_feature.py              # Sequence to features converter
├── build_nacid_encoder_data.py    # Main dataset builder
└── build_encoder_dataset.sh       # Pipeline execution script
```

## Output Location

Generated datasets are saved to:
- DNA: `data/encoder/nacid/dna_encoder.parquet`
- RNA: `data/encoder/nacid/rna_encoder.parquet`
- Test outputs: `data/encoder/nacid/*_test.parquet`

## Checkpointing

The pipeline supports automatic checkpointing:
- Saves progress every 1000 sequences (configurable)
- Can resume from checkpoint if interrupted
- Use `--no_resume` to start fresh

## Performance Notes

- **Processing time**: ~0.5-2 seconds per sequence (depends on length)
- **Sequence length**: Sequences < 10 or > 1000 nucleotides are skipped
- **Parallelization**: Scales well with multiple workers (8-16 recommended)
- **Memory**: Temporary PDB files are cleaned up after each sequence

## Troubleshooting

**Error: 'fiber' command not found**
- Install X3DNA: http://x3dna.org/
- Add to PATH: `export PATH=/path/to/x3dna/bin:$PATH`
- Or set: `export X3DNA_FIBER=/path/to/x3dna/bin/fiber`

**Sequences failing to process**
- Check sequence format (DNA: ACGT, RNA: ACGU)
- Sequences with ambiguous bases may fail
- Very long sequences (>1000) are skipped by default

**Out of memory**
- Reduce `--num_workers` or `--batch_size`
- Process datasets separately (DNA and RNA)

## Related Files

- **3D structure tools**: `examples/3dna_test.py`
- **Protein pipeline**: `src/data_factory/protein/`
- **Data structure docs**: `docs/encoder_data_structure.md`

