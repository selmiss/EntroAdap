#!/usr/bin/env python3
"""
Profile the sequence processing pipeline to identify bottlenecks.
"""

import sys
import time
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data_factory.nacid.seq_to_feature import sequence_to_features


def profile_single_sequence(seq: str, seq_type: str = "dna"):
    """Profile processing of a single sequence."""
    
    print("=" * 70)
    print(f"Profiling {seq_type.upper()} sequence processing")
    print(f"Sequence length: {len(seq)}")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        timings = {}
        
        # Total time
        t_start_total = time.time()
        
        result = sequence_to_features(
            seq=seq,
            seq_id="profile_test",
            seq_type=seq_type,
            workdir=tmpdir,
            fiber_exe="fiber"
        )
        
        t_total = time.time() - t_start_total
        
        if result:
            print(f"\n✓ Processing successful")
            print(f"  Sequence length: {result['seq_length']}")
            print(f"  Number of atoms: {result['num_atoms']}")
            print(f"  Number of edges: {result['edge_index'].shape[1]}")
            print(f"\n⏱️  Total time: {t_total:.3f} seconds")
            print(f"  Time per nucleotide: {t_total/len(seq)*1000:.2f} ms")
            print(f"  Time per atom: {t_total/result['num_atoms']*1000:.2f} ms")
        else:
            print(f"✗ Processing failed")
            print(f"⏱️  Time: {t_total:.3f} seconds")
        
        return t_total


def profile_with_detailed_timing(seq: str, seq_type: str = "dna"):
    """Profile with detailed step-by-step timing."""
    import os
    import subprocess
    import numpy as np
    from pathlib import Path
    
    # Import internal functions
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "x3dna_test", 
        str(project_root / "examples" / "3dna_test.py")
    )
    x3dna_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x3dna_test)
    
    from src.data_factory.nacid.seq_to_feature import (
        parse_pdb_atoms, atom_info_to_features, build_radius_graph
    )
    
    print("\n" + "=" * 70)
    print("Detailed timing breakdown")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        timings = {}
        
        # Step 1: X3DNA fiber structure generation
        t_start = time.time()
        try:
            result = x3dna_test.nucleic_acid_to_coords_or_cif(
                seq=seq[:500],  # Truncate like the real pipeline
                seq_type=seq_type,
                workdir=tmpdir,
                write_cif=False,
                fiber_exe="fiber"
            )
            pdb_path = result['pdb_path']
            timings['1_fiber'] = time.time() - t_start
        except Exception as e:
            print(f"✗ Fiber failed: {e}")
            return None
        
        # Step 2: Parse PDB
        t_start = time.time()
        try:
            atom_info_list, coordinates = parse_pdb_atoms(pdb_path)
            timings['2_parse_pdb'] = time.time() - t_start
        except Exception as e:
            print(f"✗ PDB parsing failed: {e}")
            return None
        
        # Step 3: Extract features
        t_start = time.time()
        try:
            node_feat = atom_info_to_features(atom_info_list)
            timings['3_features'] = time.time() - t_start
        except Exception as e:
            print(f"✗ Feature extraction failed: {e}")
            return None
        
        # Step 4: Build graph
        t_start = time.time()
        try:
            edge_index, edge_attr = build_radius_graph(
                coordinates,
                radius=8.0,
                max_neighbors=24,
                sym_mode="union"
            )
            timings['4_graph'] = time.time() - t_start
        except Exception as e:
            print(f"✗ Graph building failed: {e}")
            return None
        
        # Print breakdown
        total = sum(timings.values())
        print(f"\nStep-by-step timing:")
        for step, t in timings.items():
            pct = (t / total * 100) if total > 0 else 0
            print(f"  {step:20s}: {t:6.3f}s ({pct:5.1f}%)")
        print(f"  {'TOTAL':20s}: {total:6.3f}s")
        
        # Identify bottleneck
        bottleneck = max(timings.items(), key=lambda x: x[1])
        print(f"\n🔍 Bottleneck: {bottleneck[0]} ({bottleneck[1]:.3f}s, {bottleneck[1]/total*100:.1f}%)")
        
        return timings


if __name__ == "__main__":
    import os
    
    fiber_exe = os.environ.get("X3DNA_FIBER", "fiber")
    
    # Test sequences of different lengths
    test_sequences = {
        "short_50": "A" * 50 + "C" * 50,
        "medium_200": "ATCG" * 50,
        "long_500": "ATCGATCGATCG" * 42,  # ~500 nt
    }
    
    print("Testing DNA sequences of different lengths...")
    print()
    
    for name, seq in test_sequences.items():
        print(f"\n{'='*70}")
        print(f"Testing {name} (length: {len(seq)})")
        print(f"{'='*70}")
        t = profile_single_sequence(seq, "dna")
        
        if t:
            # Detailed timing for this sequence
            profile_with_detailed_timing(seq, "dna")
    
    # Summary and recommendations
    print("\n" + "=" * 70)
    print("Performance Analysis Summary")
    print("=" * 70)
    print("\n💡 Recommendations:")
    print("  1. If X3DNA fiber is the bottleneck:")
    print("     - Reduce max sequence length (currently 500)")
    print("     - Cannot parallelize fiber itself (CLI tool)")
    print("  2. If graph building is slow:")
    print("     - Reduce max_neighbors (currently 24)")
    print("     - Reduce radius (currently 8.0Å)")
    print("  3. Overall optimization:")
    print("     - Increase batch_size (currently 50)")
    print("     - Increase num_workers (currently 8)")
    print("     - Consider filtering very long sequences")

