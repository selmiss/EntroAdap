#!/usr/bin/env python3
"""
Build molecule QA benchmark dataset from DQ-Former data.
Converts JSONL format to parquet with structural features.
Supports single or multiple molecules per sample.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any
import argparse
from multiprocessing import Pool, cpu_count

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.data_factory.molecule.mol_structure import generate_2d_3d_from_smiles


def convert_conversations_to_messages(system: str, conversations: list) -> list:
    """Convert DQ-Former conversation format to messages format."""
    messages = []
    
    # Add system message if present
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    
    # Convert conversations
    for conv in conversations:
        if "user" in conv:
            messages.append({"role": "user", "content": conv["user"]})
        if "assistant" in conv:
            messages.append({"role": "assistant", "content": conv["assistant"]})
    
    return messages


def process_molecule_qa(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a QA sample with optional SMILES (single or multiple) into dataset format."""
    smiles_raw = row.get('smiles')
    
    messages = convert_conversations_to_messages(
        row.get('system', ''),
        row.get('conversations', [])
    )
    
    if not messages:
        return None
    
    if not smiles_raw:
        data = {
            'modality': 'text',
            'messages': messages,
        }
        if 'cid' in row:
            data['cid'] = row['cid']
        if 'category' in row:
            data['category'] = row['category']
        return data
    
    # Normalize to list format
    smiles_list = smiles_raw if isinstance(smiles_raw, list) else [smiles_raw]
    smiles_list = [s for s in smiles_list if s and s.strip()]
    
    if not smiles_list:
        return None
    
    try:
        # Process each molecule
        entities = []
        for smiles in smiles_list:
            atoms, graph_2d, coordinates_3d = generate_2d_3d_from_smiles(smiles)
            
            if atoms is None or graph_2d is None or coordinates_3d is None:
                return None
            
            entity = {
                'node_feat': graph_2d['node_feat'].numpy().tolist(),
                'pos': coordinates_3d.tolist(),
                'edge_index': graph_2d['edge_index'].numpy().tolist(),
                'chem_edge_index': graph_2d['chem_edge_index'].numpy().tolist(),
                'chem_edge_feat_cat': graph_2d['chem_edge_feat_cat'].numpy().tolist(),
            }
            
            if 'edge_feat_dist' in graph_2d:
                entity['edge_feat_dist'] = graph_2d['edge_feat_dist'].numpy().tolist()
            
            entities.append(entity)
        
        # Build dataset entry (always use list format for consistency)
        data = {
            'modality': 'molecule',
            'smiles': smiles_list,
            'cid': row.get('cid', ''),
            'category': row.get('category', ''),
            'messages': messages,
            'node_feat': [e['node_feat'] for e in entities],
            'pos': [e['pos'] for e in entities],
            'edge_index': [e['edge_index'] for e in entities],
            'chem_edge_index': [e['chem_edge_index'] for e in entities],
            'chem_edge_feat_cat': [e['chem_edge_feat_cat'] for e in entities],
        }
        
        if 'edge_feat_dist' in entities[0]:
            data['edge_feat_dist'] = [e['edge_feat_dist'] for e in entities]
        
        return data
    
    except Exception as e:
        return None


def process_batch(rows: list) -> list:
    """Process a batch of rows for multiprocessing."""
    return [process_molecule_qa(row) for row in rows]


def process_split(
    input_file: Path,
    output_file: Path,
    num_workers: int = None,
    max_workers: int = None,
    batch_size: int = 100,
    limit: int = None,
    verbose: bool = True
):
    """Process a single JSONL file."""
    if verbose:
        print(f"\nProcessing {input_file.name}...")
    
    # Read JSONL file
    data_list = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data_list.append(json.loads(line))
    
    if verbose:
        print(f"Loaded {len(data_list)} samples")
    
    # Determine number of workers
    if num_workers is None:
        if limit and limit <= 50:
            num_workers = min(8, cpu_count())
        else:
            num_workers = max(1, cpu_count() - 1)
    
    # Apply max_workers limit
    if max_workers is not None:
        num_workers = min(num_workers, max_workers)
    
    if verbose:
        print(f"Using {num_workers} workers with batch size {batch_size}")
    
    # Process in parallel
    results = []
    success_count = 0
    fail_count = 0
    
    if num_workers > 1:
        batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
        
        with Pool(num_workers) as pool:
            for batch_results in tqdm(
                pool.imap(process_batch, batches),
                total=len(batches),
                desc=f"Processing {input_file.name}",
                disable=not verbose
            ):
                for data in batch_results:
                    if data is not None:
                        results.append(data)
                        success_count += 1
                    else:
                        fail_count += 1
    else:
        for row in tqdm(data_list, desc=f"Processing {input_file.name}", disable=not verbose):
            data = process_molecule_qa(row)
            if data is not None:
                results.append(data)
                success_count += 1
            else:
                fail_count += 1
    
    # Save to parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_parquet(output_file, index=False)
    
    if verbose:
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"Successfully processed: {success_count}/{len(data_list)} samples")
        if fail_count > 0:
            print(f"Failed: {fail_count} samples")
        print(f"Saved to: {output_file}")
        print(f"File size: {file_size_mb:.2f} MB")
    
    return df


def build_mol_qa_dataset(
    input_dir: str,
    output_dir: str,
    num_workers: int = None,
    max_workers: int = None,
    batch_size: int = 100,
    limit: int = None,
    verbose: bool = True
):
    """Build molecule QA dataset from JSONL files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if verbose:
        print("=" * 60)
        print("Building Molecule QA Benchmark Dataset")
        if limit:
            print(f"DEMO MODE: Processing first {limit} samples only")
        print("=" * 60)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
    
    results = {}
    splits = ['train', 'test']
    
    for split in splits:
        input_file = input_dir / f"{split}.jsonl"
        output_file = output_dir / f"{split}.parquet" if not limit else output_dir / f"{split}_demo.parquet"
        
        if not input_file.exists():
            print(f"\nWarning: {input_file} not found, skipping...")
            continue
        
        df = process_split(input_file, output_file, num_workers, max_workers, batch_size, limit, verbose)
        results[split] = df
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        for split, df in results.items():
            print(f"{split.capitalize():6s}: {len(df):,} samples")
        print("=" * 60)
        
        # Print first sample
        if results:
            first_split = list(results.keys())[0]
            df = results[first_split]
            if len(df) > 0:
                print(f"\nFirst sample from {first_split} split:")
                first = df.iloc[0]
                print(f"  Modality: {first['modality']}")
                if 'smiles' in first:
                    smiles_val = first['smiles']
                    num_mols = len(smiles_val) if isinstance(smiles_val, list) else 1
                    print(f"  SMILES: {smiles_val if not isinstance(smiles_val, list) or num_mols == 1 else f'{num_mols} molecules'}")
                if 'cid' in first:
                    print(f"  CID: {first['cid']}")
                if 'category' in first:
                    print(f"  Category: {first['category']}")
                print(f"  Messages: {len(first['messages'])} messages")
                
                # Print structure data if molecule
                if first['modality'] == 'molecule':
                    print(f"\n  Structure data:")
                    for key in ['node_feat', 'pos', 'edge_index', 'chem_edge_index']:
                        if key in first:
                            val = first[key]
                            # All molecules now stored in list format
                            if isinstance(val, list) and len(val) > 0:
                                num_entities = len(val)
                                entity_desc = f"{num_entities} entity" if num_entities == 1 else f"{num_entities} entities"
                                print(f"    {key:20s}: list format ({entity_desc})")
                            else:
                                print(f"    {key:20s}: {type(val)}")
    
    print("\n✅ Molecule QA dataset building complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Build QA benchmark dataset from JSONL files (supports single/multiple molecules per sample)"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/home/UWO/zjing29/proj/DQ-Former/data/mol_qa',
        help='Directory containing JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/benchmark/mol_qa',
        help='Output directory for parquet files'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto-detect)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help='Maximum number of workers to use (caps auto-detected value)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size for multiprocessing'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process only first N samples (for demo/testing)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    build_mol_qa_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        limit=args.limit,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

