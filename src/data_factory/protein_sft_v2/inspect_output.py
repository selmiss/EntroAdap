#!/usr/bin/env python3
"""
Inspect and validate protein SFT dataset output (Parquet format).
"""

import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


def load_parquet_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from Parquet file(s).
    
    Args:
        path: Path to Parquet file or directory containing Parquet files
        
    Returns:
        DataFrame with all records
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    if path_obj.is_file():
        # Single file
        return pd.read_parquet(path_obj)
    elif path_obj.is_dir():
        # Directory with multiple files
        parquet_files = sorted(path_obj.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No Parquet files found in directory: {path}")
        
        print(f"Found {len(parquet_files)} Parquet file(s):")
        for f in parquet_files:
            file_size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({file_size_mb:.2f} MB)")
        print()
        
        # Load all files
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Path is neither a file nor a directory: {path}")


def inspect_dataset(path: str, show_instructions: bool = False, max_display: int = 5):
    """
    Inspect a protein SFT dataset from Parquet file(s).
    
    Args:
        path: Path to Parquet file or directory containing Parquet files
        show_instructions: Whether to display full instructions
        max_display: Maximum number of records to display in detail
    """
    # Load dataset
    df = load_parquet_dataset(path)
    records = df.to_dict('records')
    
    if not records:
        print("No records found in dataset!")
        return
    
    print("=" * 80)
    print(f"PROTEIN SFT DATASET INSPECTION")
    print("=" * 80)
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Total records: {len(records)}")
    
    # Field presence
    print(f"\n✓ Field coverage:")
    required_fields = ['uniprot_id', 'pdb_id', 'sequence', 'structure', 'instruction', 'modality']
    for field in required_fields:
        present = sum(1 for r in records if field in r and r[field] is not None)
        print(f"  {field}: {present}/{len(records)} ({present/len(records)*100:.1f}%)")
    
    # Sequence statistics
    seq_lengths = [len(r['sequence']) for r in records if 'sequence' in r and r['sequence']]
    if seq_lengths:
        print(f"\n📏 Sequence length:")
        print(f"  Mean: {sum(seq_lengths)/len(seq_lengths):.1f} aa")
        print(f"  Min: {min(seq_lengths)} aa")
        print(f"  Max: {max(seq_lengths)} aa")
    
    # Structure statistics
    num_atoms_list = []
    for r in records:
        if 'structure' in r and r['structure'] is not None:
            if isinstance(r['structure'], dict):
                num_atoms_list.append(r['structure'].get('num_atoms', 0))
    
    if num_atoms_list:
        print(f"\n🧬 Structure (atoms):")
        print(f"  Mean: {sum(num_atoms_list)/len(num_atoms_list):.1f}")
        print(f"  Min: {min(num_atoms_list)}")
        print(f"  Max: {max(num_atoms_list)}")
    
    # Instruction statistics
    instruction_lengths = [len(str(r['instruction'])) for r in records if 'instruction' in r and r['instruction']]
    if instruction_lengths:
        print(f"\n📝 Instruction length:")
        print(f"  Mean: {sum(instruction_lengths)/len(instruction_lengths):.1f} chars")
        print(f"  Min: {min(instruction_lengths)} chars")
        print(f"  Max: {max(instruction_lengths)} chars")
    
    # Method distribution
    methods = {}
    for r in records:
        method = r.get('method', 'Unknown')
        methods[method] = methods.get(method, 0) + 1
    
    print(f"\n🔬 Experimental methods:")
    for method, count in sorted(methods.items(), key=lambda x: -x[1])[:10]:
        print(f"  {method}: {count} ({count/len(records)*100:.1f}%)")
    
    # Resolution statistics (for X-ray and EM)
    resolutions = []
    for r in records:
        res_str = r.get('resolution', '')
        if res_str and res_str != '-' and res_str != 'None':
            try:
                # Extract numeric value (e.g., "1.90 A" -> 1.90)
                res_val = float(str(res_str).split()[0])
                resolutions.append(res_val)
            except:
                pass
    
    if resolutions:
        print(f"\n🔍 Resolution (for structures with resolution):")
        print(f"  Mean: {sum(resolutions)/len(resolutions):.2f} Å")
        print(f"  Min: {min(resolutions):.2f} Å")
        print(f"  Max: {max(resolutions):.2f} Å")
    
    # Display detailed records
    if show_instructions and max_display > 0:
        print(f"\n{'='*80}")
        print(f"DETAILED RECORDS (showing {min(max_display, len(records))})")
        print(f"{'='*80}")
        
        for i, record in enumerate(records[:max_display]):
            print(f"\n{'='*80}")
            print(f"RECORD {i+1}/{len(records)}")
            print(f"{'='*80}")
            print(f"UniProt ID: {record.get('uniprot_id', 'N/A')}")
            print(f"PDB ID: {record.get('pdb_id', 'N/A')}")
            print(f"Method: {record.get('method', 'N/A')}")
            print(f"Resolution: {record.get('resolution', 'N/A')}")
            print(f"Chains: {record.get('chains', 'N/A')}")
            print(f"Sequence length: {len(record.get('sequence', '')) if record.get('sequence') else 0} aa")
            
            if 'structure' in record and record['structure'] is not None:
                if isinstance(record['structure'], dict):
                    print(f"Number of atoms: {record['structure'].get('num_atoms', 'N/A')}")
                    if 'edge_index' in record['structure'] and record['structure']['edge_index'] is not None:
                        edge_idx = record['structure']['edge_index']
                        if isinstance(edge_idx, list) and len(edge_idx) > 0:
                            if isinstance(edge_idx[0], list):
                                print(f"Number of edges: {len(edge_idx[0])}")
                            else:
                                print(f"Number of edges: N/A (unexpected format)")
            
            print(f"\nInstruction:")
            print("-" * 80)
            print(record.get('instruction', 'N/A'))
            print("-" * 80)
    
    # Validation checks
    print(f"\n{'='*80}")
    print("VALIDATION CHECKS")
    print(f"{'='*80}")
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: All records have required fields
    total_checks += 1
    all_have_required = all(
        all(field in r and r[field] is not None for field in required_fields)
        for r in records
    )
    if all_have_required:
        print("✅ All records have required fields")
        checks_passed += 1
    else:
        print("❌ Some records missing required fields")
    
    # Check 2: All structures have valid data
    total_checks += 1
    all_valid_structure = True
    for r in records:
        if 'structure' not in r or r['structure'] is None:
            all_valid_structure = False
            break
        if isinstance(r['structure'], dict):
            if 'num_atoms' not in r['structure'] or r['structure']['num_atoms'] <= 0:
                all_valid_structure = False
                break
            if 'node_feat' not in r['structure'] or 'coordinates' not in r['structure'] or 'edge_index' not in r['structure']:
                all_valid_structure = False
                break
    
    if all_valid_structure:
        print("✅ All structures have valid data")
        checks_passed += 1
    else:
        print("❌ Some structures have invalid data")
    
    # Check 3: All instructions are non-empty
    total_checks += 1
    empty_count = sum(1 for r in records if not r.get('instruction') or len(str(r.get('instruction', '')).strip()) == 0)
    if empty_count == 0:
        print("✅ All records have non-empty instructions")
        checks_passed += 1
    else:
        print(f"❌ {empty_count} records have empty instructions")
    
    # Check 4: Modality field is correct
    total_checks += 1
    all_protein_modality = all(r.get('modality') == 'protein' for r in records)
    if all_protein_modality:
        print("✅ All records have modality='protein'")
        checks_passed += 1
    else:
        print("❌ Some records have incorrect modality")
    
    print(f"\nValidation score: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("\n🎉 Dataset is valid and ready for use!")
    else:
        print("\n⚠️  Dataset has some issues that should be addressed.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_output.py <parquet_file_or_directory> [--show-instructions] [--max-display N]")
        print("\nOptions:")
        print("  --show-instructions: Display full instruction text for each record")
        print("  --max-display N: Maximum number of records to display (default: 5)")
        print("\nExamples:")
        print("  python inspect_output.py data/sft/protein")
        print("  python inspect_output.py data/sft/protein/protein_sft.parquet --show-instructions")
        sys.exit(1)
    
    path = sys.argv[1]
    show_instructions = '--show-instructions' in sys.argv
    
    max_display = 5
    if '--max-display' in sys.argv:
        idx = sys.argv.index('--max-display')
        if idx + 1 < len(sys.argv):
            max_display = int(sys.argv[idx + 1])
    
    try:
        inspect_dataset(path, show_instructions, max_display)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
