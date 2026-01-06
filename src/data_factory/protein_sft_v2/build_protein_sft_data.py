#!/usr/bin/env python3
"""
Build protein SFT (Supervised Fine-Tuning) dataset from UniProt JSON files.

This script:
1. Scans UniProt JSON files from data/uniprot/full
2. Filters proteins with exactly one PDB ID
3. Extracts structural features (coordinates, atom info)
4. Extracts text comments/descriptions
5. Constructs fluent instructions using OpenAI API
6. Saves combined dataset in Parquet format with automatic file splitting
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data_factory.protein.pdbid_to_feature import pdbid_to_features
from src.data_factory.protein.map_fetch_pdb3d import download_pdb_structures
from utils.gpt_helper.openai_api import chat_completion, run_batch_requests


# System prompt for instruction construction (from instruction_construction.py)
SYSTEM_PROMPT = """You are an expert in protein biology and scientific communication. 
Your task is to convert structured protein annotation comments into fluent, well-structured instructional text.

The input will be a list of comment types and values from protein databases. Your output should be a smooth, coherent instruction or description that naturally integrates all the information provided.

Guidelines:
- Create a natural, flowing narrative from the structured comments
- Maintain scientific accuracy while improving readability
- Organize information logically (e.g., function -> structure -> location -> regulation)
- Use clear transitions between different aspects
- Be concise but comprehensive
- Use proper scientific terminology
- Output ONLY the instruction text, without any meta-commentary or formatting markers"""


def extract_protein_records_from_uniprot(json_file: Path) -> List[Dict[str, Any]]:
    """
    Extract protein records with exactly one PDB ID from UniProt JSON file.
    
    Args:
        json_file: Path to UniProt JSON file
        
    Returns:
        List of dicts with protein info (uniprot_id, pdb_id, comments, sequence, etc.)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    protein_records = []
    for entry in data.get('results', []):
        uniprot_id = entry.get('primaryAccession')
        if not uniprot_id:
            continue
        
        # Extract PDB cross-references
        pdb_refs = []
        for ref in entry.get('uniProtKBCrossReferences', []):
            if ref.get('database') == 'PDB':
                pdb_id = ref.get('id')
                if not pdb_id:
                    continue
                
                # Extract properties (method, resolution, chains)
                props = {}
                for prop in ref.get('properties', []):
                    props[prop.get('key')] = prop.get('value')
                
                pdb_refs.append({
                    'pdb_id': pdb_id.lower(),  # Normalize to lowercase
                    'method': props.get('Method', ''),
                    'resolution': props.get('Resolution', ''),
                    'chains': props.get('Chains', ''),
                })
        
        # Filter: only keep proteins with exactly one PDB ID
        if len(pdb_refs) != 1:
            continue
        
        # Extract comments (following uniprot_fitler.py logic)
        comments = []
        for comment in entry.get('comments', []):
            comment_type = comment.get('commentType', '')
            
            if 'texts' in comment:
                for text_obj in comment['texts']:
                    if 'value' in text_obj:
                        comments.append({
                            'type': comment_type,
                            'value': text_obj['value']
                        })
            
            if 'note' in comment and 'texts' in comment['note']:
                for text_obj in comment['note']['texts']:
                    if 'value' in text_obj:
                        comments.append({
                            'type': f"{comment_type} (note)",
                            'value': text_obj['value']
                        })
        
        # Extract sequence
        sequence = entry.get('sequence', {}).get('value', '')
        
        # Build protein record
        pdb_ref = pdb_refs[0]
        protein_records.append({
            'uniprot_id': uniprot_id,
            'pdb_id': pdb_ref['pdb_id'],
            'method': pdb_ref['method'],
            'resolution': pdb_ref['resolution'],
            'chains': pdb_ref['chains'],
            'sequence': sequence,
            'comments': comments,
        })
    
    return protein_records


def create_user_prompt(comments: List[Dict[str, str]]) -> str:
    """
    Create a user prompt from a list of comment dictionaries.
    (From instruction_construction.py)
    
    Args:
        comments: List of dicts with 'type' and 'value' keys
        
    Returns:
        Formatted prompt string
    """
    if not comments:
        return "No comments available."
    
    prompt_lines = ["Convert the following protein annotations into a fluent instruction:\n"]
    for comment in comments:
        comment_type = comment.get("type", "UNKNOWN")
        comment_value = comment.get("value", "")
        prompt_lines.append(f"[{comment_type}] {comment_value}")
    
    return "\n".join(prompt_lines)


def process_structure(
    pdb_id: str,
    structure_dir: Path,
    ca_only: bool = False,
    graph_radius: float = 8.0,
    max_neighbors: int = 24,
    max_atoms: int = 30000
) -> Optional[Dict[str, Any]]:
    """
    Process a single PDB structure and extract features.
    (Based on build_protein_encoder_data.py)
    
    Args:
        pdb_id: PDB ID (lowercase)
        structure_dir: Directory containing CIF files
        ca_only: Extract only C-alpha atoms (default: False for all atoms)
        graph_radius: Radius for graph construction
        max_neighbors: Max neighbors per node
        max_atoms: Maximum number of atoms allowed (default: 30000)
        
    Returns:
        Dictionary with structural features, or None if processing fails
    """
    try:
        # Generate features using existing function
        data = pdbid_to_features(
            pdb_id,
            structure_dir=str(structure_dir),
            graph_radius=graph_radius,
            ca_only=ca_only,
            max_neighbors=max_neighbors,
            sym_mode="union"
        )
        
        if data is None:
            return None
        
        # Check atom count threshold
        num_atoms = data['num_nodes']
        if num_atoms > max_atoms:
            # Skip structures that are too large
            return None
        
        # Convert to list format for JSON storage
        result = {
            'num_atoms': num_atoms,
            'node_feat': data['node_feat'].tolist(),
            'coordinates': data['coordinates'].tolist(),
            'edge_index': data['edge_index'].tolist(),
            'edge_attr': data['edge_attr'].tolist(),
        }
        
        return result
    
    except Exception as e:
        # Silently skip structures that fail to process (CIF parsing errors, etc.)
        return None


def build_protein_sft_dataset(
    uniprot_json_dir: str,
    structure_dir: str,
    output_dir: str,
    ca_only: bool = False,
    graph_radius: float = 8.0,
    max_neighbors: int = 24,
    max_atoms: int = 30000,
    download_missing: bool = True,
    download_delay: float = 0.1,
    use_batch_api: bool = True,
    model: str = "gpt-5-mini",
    max_samples: Optional[int] = None,
    max_records_per_file: int = 1000,
    verbose: bool = True,
):
    """
    Build protein SFT dataset from UniProt JSON files.
    
    Args:
        uniprot_json_dir: Directory containing UniProt JSON files
        structure_dir: Directory to store/read PDB CIF files
        output_dir: Output directory for Parquet files
        ca_only: Extract only C-alpha atoms (default: False for all atoms)
        graph_radius: Radius for graph construction (default: 8.0)
        max_neighbors: Max neighbors per node (default: 24)
        max_atoms: Maximum atoms allowed per structure (default: 30000)
        download_missing: Download missing PDB structures
        download_delay: Delay between download requests in seconds (default: 0.1)
        use_batch_api: Use batch API (True) or sequential API (False) for OpenAI
        model: OpenAI model to use (default: gpt-5-mini)
        max_samples: Maximum number of samples to process (for testing)
        max_records_per_file: Maximum records per Parquet file (default: 1000)
        verbose: Print progress information
    """
    uniprot_dir = Path(uniprot_json_dir)
    struct_dir = Path(structure_dir)
    output_path = Path(output_dir)
    
    if verbose:
        print("=" * 70)
        print("Building Protein SFT Dataset")
        print("=" * 70)
        print(f"UniProt JSON directory: {uniprot_dir}")
        print(f"Structure directory: {struct_dir}")
        print(f"Output directory: {output_path}")
        print(f"Atom selection: {'C-alpha only' if ca_only else 'All atoms'}")
        print(f"Max atoms per structure: {max_atoms}")
        print(f"Graph radius: {graph_radius} Å")
        print(f"Max neighbors: {max_neighbors}")
        print(f"OpenAI model: {model}")
        print(f"API mode: {'Batch' if use_batch_api else 'Sequential'}")
        print(f"Max records per file: {max_records_per_file}")
    
    # Step 1: Extract protein records from UniProt JSON files
    if verbose:
        print("\n" + "=" * 70)
        print("Step 1: Extracting protein records from UniProt JSON files")
        print("=" * 70)
    
    json_files = sorted(uniprot_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {uniprot_dir}")
    
    if verbose:
        print(f"Found {len(json_files)} JSON files:")
        for f in json_files:
            print(f"  - {f.name}")
    
    all_protein_records = []
    for json_file in tqdm(json_files, desc="Reading JSON files", disable=not verbose):
        records = extract_protein_records_from_uniprot(json_file)
        all_protein_records.extend(records)
    
    if verbose:
        print(f"\nTotal protein records with single PDB: {len(all_protein_records)}")
    
    # Limit for testing if requested
    if max_samples and max_samples < len(all_protein_records):
        all_protein_records = all_protein_records[:max_samples]
        if verbose:
            print(f"Limited to {max_samples} samples for testing")
    
    if len(all_protein_records) == 0:
        print("No protein records found with single PDB ID!")
        return
    
    # Step 2: Download missing PDB structures
    if download_missing:
        if verbose:
            print("\n" + "=" * 70)
            print("Step 2: Downloading missing PDB structures")
            print("=" * 70)
        
        struct_dir.mkdir(parents=True, exist_ok=True)
        
        # Check which structures are missing
        missing_pdb_ids = []
        for record in all_protein_records:
            pdb_id = record['pdb_id']
            cif_path = struct_dir / f"{pdb_id}.cif"
            
            if not cif_path.exists():
                missing_pdb_ids.append(pdb_id)
        
        # Deduplicate
        missing_pdb_ids = list(set(missing_pdb_ids))
        
        if verbose:
            print(f"Missing structures: {len(missing_pdb_ids)}/{len(set(r['pdb_id'] for r in all_protein_records))}")
        
        if missing_pdb_ids:
            if verbose:
                print("Downloading missing structures...")
                print(f"Rate limiting: {download_delay}s delay between requests (~{1/download_delay:.1f} req/sec)")
            download_pdb_structures(
                missing_pdb_ids, 
                str(struct_dir), 
                file_format='cif',
                delay=download_delay,
                verbose=verbose
            )
    
    # Step 3: Process structures and add structural features
    if verbose:
        print("\n" + "=" * 70)
        print("Step 3: Processing PDB structures and extracting features")
        print("=" * 70)
    
    records_with_structure = []
    structure_success = 0
    structure_failed = 0
    structure_too_large = 0
    
    for record in tqdm(all_protein_records, desc="Processing structures", disable=not verbose):
        pdb_id = record['pdb_id']
        
        # Extract structural features with exception handling
        try:
            structure_data = process_structure(
                pdb_id, struct_dir, ca_only, graph_radius, max_neighbors, max_atoms
            )
            
            if structure_data is not None:
                # Check if structure was rejected due to size
                if structure_data['num_atoms'] <= max_atoms:
                    # Add structure data to record
                    record['structure'] = structure_data
                    records_with_structure.append(record)
                    structure_success += 1
                else:
                    structure_too_large += 1
            else:
                structure_failed += 1
        except Exception as e:
            # Catch any unexpected errors in structure processing
            if verbose:
                tqdm.write(f"Error processing {pdb_id}: {str(e)}")
            structure_failed += 1
    
    if verbose:
        print(f"\nStructure processing results:")
        print(f"  Success: {structure_success}/{len(all_protein_records)}")
        print(f"  Failed (CIF parsing/other): {structure_failed}/{len(all_protein_records)}")
        if structure_too_large > 0:
            print(f"  Skipped (too large, >{max_atoms} atoms): {structure_too_large}/{len(all_protein_records)}")
    
    if len(records_with_structure) == 0:
        print("No records with valid structure data!")
        return
    
    # Step 4: Generate instructions using OpenAI API
    if verbose:
        print("\n" + "=" * 70)
        print("Step 4: Generating instructions using OpenAI API")
        print("=" * 70)
    
    # Create prompts only for records with valid structures
    # This saves money by not sending API requests for failed structures
    requests = []
    valid_records = []
    
    for record in records_with_structure:
        comments = record.get('comments', [])
        # Only send to API if we have comments (avoid wasting money on empty prompts)
        if comments:
            user_prompt = create_user_prompt(comments)
            requests.append((SYSTEM_PROMPT, user_prompt))
            valid_records.append(record)
        else:
            if verbose:
                tqdm.write(f"Warning: No comments for {record.get('uniprot_id')}, skipping API call")
    
    if verbose:
        print(f"Generating instructions for {len(requests)} proteins (with valid comments)...")
        if len(requests) < len(records_with_structure):
            print(f"  Skipped {len(records_with_structure) - len(requests)} proteins with no comments")
    
    # Call OpenAI API with robust error handling
    if use_batch_api:
        if verbose:
            print("Using batch API (this may take some time)...")
        try:
            # Note: Sending all requests in a single batch
            # For very large datasets, consider splitting into smaller batches
            responses = run_batch_requests(
                requests,
                model=model,
                poll_interval=5.0,
                completion_window="24h",
            )
        except Exception as e:
            if verbose:
                print(f"Error in batch API call: {e}")
                print("Falling back to sequential API...")
            # Fallback to sequential if batch fails
            responses = []
            for i, (system_prompt, user_prompt) in enumerate(tqdm(requests, desc="API calls (fallback)", disable=not verbose)):
                try:
                    response = chat_completion(system_prompt, user_prompt, model=model)
                    responses.append(response)
                except Exception as e:
                    if verbose:
                        tqdm.write(f"Warning: API call failed for record {i}: {e}")
                    responses.append(None)
    else:
        if verbose:
            print("Using sequential API...")
        responses = []
        for i, (system_prompt, user_prompt) in enumerate(tqdm(requests, desc="API calls", disable=not verbose)):
            try:
                response = chat_completion(system_prompt, user_prompt, model=model)
                responses.append(response)
            except Exception as e:
                if verbose:
                    tqdm.write(f"Warning: API call failed for record {i}: {e}")
                responses.append(None)
    
    if verbose:
        print(f"Received {len(responses)} responses")
        failed_api_calls = sum(1 for r in responses if r is None)
        if failed_api_calls > 0:
            print(f"  Failed API calls: {failed_api_calls}/{len(responses)}")
    
    # Update records_with_structure to only include those we sent to API
    records_with_structure = valid_records
    
    # Step 5: Combine data and save
    if verbose:
        print("\n" + "=" * 70)
        print("Step 5: Combining data and saving results")
        print("=" * 70)
    
    final_records = []
    instruction_success = 0
    instruction_failed = 0
    
    for record, instruction in zip(records_with_structure, responses):
        # Remove comments field (no longer needed)
        record.pop('comments', None)
        
        # Add instruction
        if instruction is not None and instruction.strip():
            record['instruction'] = instruction
            instruction_success += 1
        else:
            record['instruction'] = ""
            instruction_failed += 1
        
        # Add modality field
        record['modality'] = 'protein'
        
        final_records.append(record)
    
    if verbose:
        print(f"Instruction generation results:")
        print(f"  Success: {instruction_success}/{len(records_with_structure)}")
        print(f"  Failed/Empty: {instruction_failed}/{len(records_with_structure)}")
    
    # Save to Parquet files with splitting
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\nSaving to Parquet format (max {max_records_per_file} records per file)...")
    
    # Calculate number of files needed
    num_files = (len(final_records) + max_records_per_file - 1) // max_records_per_file
    
    saved_files = []
    total_size_mb = 0
    
    for file_idx in range(num_files):
        start_idx = file_idx * max_records_per_file
        end_idx = min((file_idx + 1) * max_records_per_file, len(final_records))
        batch_records = final_records[start_idx:end_idx]
        
        # Determine output filename
        if num_files == 1:
            output_file = output_path / "protein_sft.parquet"
        else:
            output_file = output_path / f"protein_sft_part{file_idx:03d}.parquet"
        
        # Convert to DataFrame and save
        df = pd.DataFrame(batch_records)
        df.to_parquet(output_file, index=False)
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        total_size_mb += file_size_mb
        saved_files.append(output_file)
        
        if verbose:
            print(f"  Saved {output_file.name}: {len(batch_records)} records, {file_size_mb:.2f} MB")
    
    if verbose:
        print(f"\n✅ Dataset building complete!")
        print(f"Total records saved: {len(final_records)}")
        print(f"Number of files: {len(saved_files)}")
        print(f"Output directory: {output_path}")
        print(f"Total size: {total_size_mb:.2f} MB")
        
        # Print sample statistics
        if final_records:
            num_atoms_list = [r['structure']['num_atoms'] for r in final_records if 'structure' in r]
            if num_atoms_list:
                print(f"\nDataset statistics:")
                print(f"  Mean atoms per structure: {sum(num_atoms_list)/len(num_atoms_list):.1f}")
                print(f"  Min atoms: {min(num_atoms_list)}")
                print(f"  Max atoms: {max(num_atoms_list)}")
    
    return saved_files


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Build protein SFT dataset from UniProt JSON files with PDB structures"
    )
    parser.add_argument(
        '--uniprot_json_dir',
        type=str,
        default='data/uniprot/full',
        help='Directory containing UniProt JSON files'
    )
    parser.add_argument(
        '--structure_dir',
        type=str,
        default='data/pdb_structures',
        help='Directory to store/read PDB CIF files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/sft/protein',
        help='Output directory for Parquet files'
    )
    parser.add_argument(
        '--max_records_per_file',
        type=int,
        default=1000,
        help='Maximum records per Parquet file (default: 1000)'
    )
    parser.add_argument(
        '--ca_only',
        action='store_true',
        help='Extract only C-alpha atoms (default: all atoms)'
    )
    parser.add_argument(
        '--max_atoms',
        type=int,
        default=30000,
        help='Maximum atoms per structure (default: 30000, structures larger than this are skipped)'
    )
    parser.add_argument(
        '--graph_radius',
        type=float,
        default=4,
        help='Radius for graph construction in Angstroms (default: 8.0)'
    )
    parser.add_argument(
        '--max_neighbors',
        type=int,
        default=16,
        help='Maximum neighbors per node (default: 24)'
    )
    parser.add_argument(
        '--no_download',
        action='store_true',
        help='Do not download missing PDB structures'
    )
    parser.add_argument(
        '--download_delay',
        type=float,
        default=0.1,
        help='Delay between PDB download requests in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--sequential_api',
        action='store_true',
        help='Use sequential API calls instead of batch API'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5-mini',
        help='OpenAI model to use (default: gpt-5-mini)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    build_protein_sft_dataset(
        uniprot_json_dir=args.uniprot_json_dir,
        structure_dir=args.structure_dir,
        output_dir=args.output_dir,
        ca_only=args.ca_only,
        graph_radius=args.graph_radius,
        max_neighbors=args.max_neighbors,
        max_atoms=args.max_atoms,
        download_missing=not args.no_download,
        download_delay=args.download_delay,
        use_batch_api=not args.sequential_api,
        model=args.model,
        max_samples=args.max_samples,
        max_records_per_file=args.max_records_per_file,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()

