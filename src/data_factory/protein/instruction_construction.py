"""Generate instruction data from UniProt protein records using OpenAI API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from tqdm import tqdm

from utils.gpt_helper.openai_api import run_batch_requests
from .cif_to_cooridinates import (
    parse_cif_atoms,
    get_ca_atoms,
    get_backbone_atoms,
)


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


def create_user_prompt(comments: List[Dict[str, str]]) -> str:
    """
    Create a user prompt from a list of comment dictionaries.
    
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


def construct_instructions_from_jsonl(
    input_file: str,
    output_file: Optional[str] = None,
    model: str = "gpt-4o-mini",
    poll_interval: float = 5.0,
    completion_window: str = "24h",
) -> List[Dict[str, Any]]:
    """
    Load a JSONL file with UniProt records, generate instruction data using OpenAI batch API,
    and save the results.
    
    Args:
        input_file: Path to input JSONL file (e.g., 'data/uniprot/test/uniprotkb_test25.jsonl')
        output_file: Optional path to output JSONL file. If None, uses input_file with '_instructions' suffix
        model: OpenAI model to use for instruction generation
        poll_interval: Seconds between batch status polls
        completion_window: Batch completion window (e.g., '24h')
        
    Returns:
        List of processed records with 'instruction' field added
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load records from JSONL
    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    if not records:
        raise ValueError(f"No records found in {input_file}")
    
    print(f"Loaded {len(records)} records from {input_file}")
    
    # Create batch requests
    requests = []
    print("Creating batch requests...")
    for record in tqdm(records, desc="Preparing prompts"):
        comments = record.get("comments", [])
        user_prompt = create_user_prompt(comments)
        requests.append((SYSTEM_PROMPT, user_prompt))
    
    print(f"\nSending {len(requests)} requests to OpenAI batch API...")
    
    # Run batch requests
    responses = run_batch_requests(
        requests,
        model=model,
        poll_interval=poll_interval,
        completion_window=completion_window,
    )
    
    print(f"Received {len(responses)} responses")
    
    # Add instructions to records and remove comments
    processed_records = []
    for i, (record, instruction) in enumerate(tqdm(zip(records, responses), total=len(records), desc="Processing responses")):
        processed_record = record.copy()
        # Remove the comments field
        processed_record.pop("comments", None)
        # Add the instruction field
        if instruction is not None:
            processed_record["instruction"] = instruction
        else:
            tqdm.write(f"Warning: No instruction generated for record {i} (uniprot_id={record.get('uniprot_id')})")
            processed_record["instruction"] = ""
        processed_records.append(processed_record)
    
    # Determine output file
    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_instructions{input_path.suffix}"
    else:
        output_path = Path(output_file)
    
    # Save to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in processed_records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    
    print(f"Saved {len(processed_records)} records to {output_path}")
    
    return processed_records


def load_id_mapping(mapping_file: str) -> Dict[str, List[str]]:
    """
    Load UniProt to PDB ID mapping from JSON file.
    
    Args:
        mapping_file: Path to mapping JSON file
        
    Returns:
        Dict mapping uniprot_id -> list of PDB IDs
    """
    mapping_path = Path(mapping_file)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Build mapping: uniprot_id -> [pdb_id1, pdb_id2, ...]
    mapping: Dict[str, List[str]] = {}
    results = data.get("results", [])
    
    for entry in results:
        uniprot_id = entry.get("from")
        pdb_id = entry.get("to")
        if uniprot_id and pdb_id:
            if uniprot_id not in mapping:
                mapping[uniprot_id] = []
            mapping[uniprot_id].append(pdb_id)
    
    print(f"Loaded mapping for {len(mapping)} UniProt IDs -> {sum(len(v) for v in mapping.values())} PDB structures")
    return mapping


def add_structure_info_from_cif(
    instruction_file: str,
    cif_directory: str,
    id_mapping_file: str,
    output_file: Optional[str] = None,
    atom_selection: Literal["all", "ca", "backbone"] = "ca",
    use_first_structure: bool = True,
) -> List[Dict[str, Any]]:
    """
    Add 3D structure information (atom info and coordinates) to instruction data from CIF files.
    
    Args:
        instruction_file: Path to JSONL file with instruction data (must have 'uniprot_id' field)
        cif_directory: Directory containing CIF files (named as {pdb_id}.cif)
        id_mapping_file: Path to JSON file mapping UniProt IDs to PDB IDs
        output_file: Optional path to output JSONL file. If None, uses instruction_file with '_with_structure' suffix
        atom_selection: Which atoms to extract - "all", "ca" (C-alpha only), or "backbone" (N, CA, C, O)
        use_first_structure: If True and multiple PDB structures exist, use only the first one.
                            If False, include all available structures as a list.
        
    Returns:
        List of records enriched with structure information
    """
    input_path = Path(instruction_file)
    cif_dir = Path(cif_directory)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
    
    if not cif_dir.exists():
        raise FileNotFoundError(f"CIF directory not found: {cif_directory}")
    
    # Load ID mapping
    uniprot_to_pdb = load_id_mapping(id_mapping_file)
    
    # Load instruction records
    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    if not records:
        raise ValueError(f"No records found in {instruction_file}")
    
    print(f"Loaded {len(records)} records from {instruction_file}")
    
    # Select appropriate parsing function
    if atom_selection == "ca":
        parse_func = get_ca_atoms
        print("Extracting C-alpha atoms only")
    elif atom_selection == "backbone":
        parse_func = get_backbone_atoms
        print("Extracting backbone atoms (N, CA, C, O)")
    else:
        parse_func = parse_cif_atoms
        print("Extracting all atoms")
    
    # Process each record and add structure info
    enriched_records = []
    success_count = 0
    failed_count = 0
    no_mapping_count = 0
    
    for i, record in enumerate(tqdm(records, desc="Adding structure info")):
        enriched_record = record.copy()
        uniprot_id = record.get("uniprot_id")
        
        if not uniprot_id:
            tqdm.write(f"Warning: Record {i} has no uniprot_id, skipping structure lookup")
            enriched_records.append(enriched_record)
            failed_count += 1
            continue
        
        # Get PDB IDs from mapping
        pdb_ids = uniprot_to_pdb.get(uniprot_id, [])
        
        if not pdb_ids:
            tqdm.write(f"Warning: No PDB mapping found for UniProt ID {uniprot_id}")
            enriched_record["structure_available"] = False
            enriched_record["pdb_ids"] = []
            enriched_records.append(enriched_record)
            no_mapping_count += 1
            failed_count += 1
            continue
        
        enriched_record["pdb_ids"] = pdb_ids
        
        # Process structure(s)
        if use_first_structure:
            # Use only the first PDB structure
            pdb_id = pdb_ids[0]
            cif_filename = f"{pdb_id}.cif"
            cif_path = cif_dir / cif_filename
            
            if not cif_path.exists():
                tqdm.write(f"Warning: CIF file not found for {uniprot_id} -> {pdb_id}: {cif_path}")
                enriched_record["structure_available"] = False
                enriched_records.append(enriched_record)
                failed_count += 1
                continue
            
            # Parse CIF file
            try:
                atom_info, coordinates, properties, failed_atoms = parse_func(str(cif_path))
                
                if atom_info:
                    enriched_record["structure_available"] = True
                    enriched_record["pdb_id"] = pdb_id
                    enriched_record["atom_info"] = atom_info
                    enriched_record["coordinates"] = coordinates
                    enriched_record["atom_properties"] = properties
                    enriched_record["num_atoms"] = len(atom_info)
                    
                    if failed_atoms:
                        enriched_record["structure_parsing_failures"] = len(failed_atoms)
                    
                    success_count += 1
                else:
                    tqdm.write(f"Warning: No atoms extracted from {cif_filename}")
                    enriched_record["structure_available"] = False
                    failed_count += 1
                    
            except Exception as e:
                tqdm.write(f"Error processing CIF for {uniprot_id} -> {pdb_id}: {e}")
                enriched_record["structure_available"] = False
                enriched_record["structure_error"] = str(e)
                failed_count += 1
        
        else:
            # Process all PDB structures
            structures = []
            has_any_structure = False
            
            for pdb_id in pdb_ids:
                cif_filename = f"{pdb_id}.cif"
                cif_path = cif_dir / cif_filename
                
                if not cif_path.exists():
                    tqdm.write(f"Warning: CIF file not found for {pdb_id}: {cif_path}")
                    continue
                
                try:
                    atom_info, coordinates, properties, failed_atoms = parse_func(str(cif_path))
                    
                    if atom_info:
                        structure_data = {
                            "pdb_id": pdb_id,
                            "atom_info": atom_info,
                            "coordinates": coordinates,
                            "atom_properties": properties,
                            "num_atoms": len(atom_info),
                        }
                        if failed_atoms:
                            structure_data["parsing_failures"] = len(failed_atoms)
                        
                        structures.append(structure_data)
                        has_any_structure = True
                    
                except Exception as e:
                    tqdm.write(f"Error processing CIF for {pdb_id}: {e}")
            
            if has_any_structure:
                enriched_record["structure_available"] = True
                enriched_record["structures"] = structures
                enriched_record["num_structures"] = len(structures)
                success_count += 1
            else:
                enriched_record["structure_available"] = False
                failed_count += 1
        
        enriched_records.append(enriched_record)
    
    print(f"\nStructure extraction summary:")
    print(f"  - Successfully added structure: {success_count}/{len(records)}")
    print(f"  - No PDB mapping found: {no_mapping_count}/{len(records)}")
    print(f"  - Failed or missing: {failed_count}/{len(records)}")
    
    # Determine output file
    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_with_structure{input_path.suffix}"
    else:
        output_path = Path(output_file)
    
    # Save enriched records
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in enriched_records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    
    print(f"Saved {len(enriched_records)} enriched records to {output_path}")
    
    return enriched_records


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python instruction_construction.py <command> [options]")
        print("\nCommands:")
        print("  construct <input_jsonl> [output_jsonl]")
        print("    - Generate instructions from UniProt comments")
        print("\n  add-structure <instruction_jsonl> <cif_directory> <id_mapping_json> [output_jsonl] [--all|--ca|--backbone] [--all-structures]")
        print("    - Add 3D structure info from CIF files to instruction data")
        print("    - Requires ID mapping file (UniProt ID -> PDB ID)")
        print("    - Use --all-structures to include all PDB structures (default: first only)")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "construct":
        if len(sys.argv) < 3:
            print("Error: construct requires input_jsonl_file")
            sys.exit(1)
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        construct_instructions_from_jsonl(input_file, output_file)
    
    elif command == "add-structure":
        if len(sys.argv) < 5:
            print("Error: add-structure requires instruction_jsonl, cif_directory, and id_mapping_json")
            sys.exit(1)
        instruction_file = sys.argv[2]
        cif_directory = sys.argv[3]
        id_mapping_file = sys.argv[4]
        output_file = None
        atom_selection = "ca"
        use_first_structure = True
        
        # Parse additional arguments
        for i in range(5, len(sys.argv)):
            arg = sys.argv[i]
            if arg in ["--all", "--ca", "--backbone"]:
                atom_selection = arg.lstrip("--")
            elif arg == "--all-structures":
                use_first_structure = False
            elif not arg.startswith("--"):
                output_file = arg
        
        add_structure_info_from_cif(
            instruction_file, 
            cif_directory, 
            id_mapping_file,
            output_file, 
            atom_selection=atom_selection,
            use_first_structure=use_first_structure
        )
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
