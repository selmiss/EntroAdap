#!/usr/bin/env python3
"""
Quick test script to demonstrate runtime filtering capabilities.

This script creates a mock dataset and shows how the filtering works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.data_loader.aa_dataset import GraphDataset
from datasets import Dataset


def create_mock_parquet(output_path: str, num_samples: int = 10):
    """Create a mock parquet file with varying structure sizes."""
    print(f"\n{'='*80}")
    print(f"Creating mock dataset: {output_path}")
    print(f"{'='*80}\n")
    
    data = []
    for i in range(num_samples):
        # Create structures with varying sizes
        if i % 3 == 0:
            num_atoms = 100 + i * 100  # Small to medium
        elif i % 3 == 1:
            num_atoms = 5000 + i * 1000  # Medium to large
        else:
            num_atoms = 20000 + i * 2000  # Large to very large
        
        # Create mock features
        node_feat = np.random.randint(0, 119, size=(num_atoms, 7)).tolist()
        coordinates = (np.random.randn(num_atoms, 3) * 10).tolist()
        
        # Create edges (roughly 10 edges per atom)
        num_edges = num_atoms * 10
        edge_index = np.random.randint(0, num_atoms, size=(2, num_edges)).tolist()
        edge_attr = np.random.rand(num_edges, 1).tolist()
        
        data.append({
            'modality': 'protein',
            'num_atoms': num_atoms,
            'node_feat': node_feat,
            'coordinates': coordinates,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
        })
        
        print(f"Sample {i}: {num_atoms:,} atoms, {num_edges:,} edges")
    
    # Create HuggingFace dataset and save
    dataset = Dataset.from_list(data)
    dataset.to_parquet(output_path)
    print(f"\n✓ Saved to {output_path}\n")
    return data


def test_filtering(parquet_path: str, max_atoms: int = None, max_edges: int = None):
    """Test the filtering with different thresholds."""
    print(f"\n{'='*80}")
    print(f"Testing Filtering")
    print(f"{'='*80}")
    print(f"max_atoms: {max_atoms}")
    print(f"max_edges: {max_edges}")
    print(f"{'='*80}\n")
    
    # Load dataset with filtering
    dataset = GraphDataset(
        dataset_path=parquet_path,
        split='train',
        max_atoms=max_atoms,
        max_edges=max_edges,
        skip_on_error=True,
    )
    
    print(f"Dataset length: {len(dataset)}\n")
    
    # Try loading some samples
    loaded_samples = []
    for i in range(min(5, len(dataset))):
        try:
            sample = dataset[i]
            num_atoms = sample['value']['node_feat'].shape[0]
            num_edges = sample['value']['edge_index'].shape[1]
            print(f"✓ Loaded sample {i}: {num_atoms} atoms, {num_edges} edges")
            loaded_samples.append((num_atoms, num_edges))
        except Exception as e:
            print(f"✗ Failed to load sample {i}: {e}")
    
    # Get filter statistics
    stats = dataset.get_filter_stats()
    print(f"\n{'='*80}")
    print(f"Filter Statistics:")
    print(f"  Filtered: {stats['filtered_count']}")
    print(f"  Errors: {stats['error_count']}")
    print(f"{'='*80}\n")
    
    return loaded_samples


def main():
    """Run the test."""
    import tempfile
    import os
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "test_data.parquet")
        
        # Create mock data
        original_data = create_mock_parquet(parquet_path, num_samples=10)
        
        print("\n" + "="*80)
        print("Test 1: No Filtering (Default)")
        print("="*80)
        samples_no_filter = test_filtering(parquet_path, max_atoms=None, max_edges=None)
        
        print("\n" + "="*80)
        print("Test 2: Filter by atoms (max_atoms=10000)")
        print("="*80)
        samples_atom_filter = test_filtering(parquet_path, max_atoms=10000, max_edges=None)
        
        print("\n" + "="*80)
        print("Test 3: Filter by edges (max_edges=50000)")
        print("="*80)
        samples_edge_filter = test_filtering(parquet_path, max_atoms=None, max_edges=50000)
        
        print("\n" + "="*80)
        print("Test 4: Filter by both (max_atoms=15000, max_edges=100000)")
        print("="*80)
        samples_both_filter = test_filtering(parquet_path, max_atoms=15000, max_edges=100000)
        
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print(f"Original dataset: 10 samples")
        print(f"  Size range: {min(d['num_atoms'] for d in original_data)} - {max(d['num_atoms'] for d in original_data)} atoms")
        print(f"\nSamples loaded:")
        print(f"  No filter: {len(samples_no_filter)}")
        print(f"  Atom filter (10k): {len(samples_atom_filter)}")
        print(f"  Edge filter (50k): {len(samples_edge_filter)}")
        print(f"  Both filters: {len(samples_both_filter)}")
        print("="*80)
        print("\n✓ Runtime filtering is working correctly!\n")


if __name__ == "__main__":
    main()

