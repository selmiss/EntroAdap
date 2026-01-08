"""
Generate mock SFT training data for proteins using OpenAI Batch API.

This script:
1. Reads protein.parquet containing structural information
2. Reads corresponding protein sequences from raw file
3. Generates diverse instruction-response pairs using OpenAI batch API
4. Outputs parquet format with messages column for SFT training

Output parquet schema:
- modality: str
- node_feat: array
- pos: array
- edge_index: array
- seq: str (original protein sequence)
- messages: list of dicts (system, user, assistant messages)

Compatible with MultiModalSFTDataset by referencing parquet rows directly.
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.gpt_helper.openai_api import run_batch_requests


# Task templates for generating diverse instructions
INSTRUCTION_TEMPLATES = [
    {
        "system": "You are a protein biochemistry expert assistant that analyzes protein structures and functions.",
        "user_template": "Analyze this protein sequence: {sequence}\n\nDescribe its likely structural features, functional domains, and potential biological roles. Be concise but informative.",
        "task_type": "structure_function_analysis"
    },
    {
        "system": "You are a structural biology expert specializing in protein folding and stability.",
        "user_template": "Examine this protein sequence: {sequence}\n\nDiscuss its secondary structure elements (alpha helices, beta sheets), potential folding patterns, and factors affecting its stability.",
        "task_type": "folding_stability"
    },
    {
        "system": "You are a molecular biology expert focusing on protein-protein interactions.",
        "user_template": "Given this protein sequence: {sequence}\n\nIdentify potential binding sites, interaction motifs, and discuss how this protein might interact with other biomolecules.",
        "task_type": "protein_interactions"
    },
    {
        "system": "You are a bioinformatics expert specializing in sequence analysis.",
        "user_template": "Analyze this protein sequence: {sequence}\n\nIdentify conserved domains, functional motifs, and predict the protein family or superfamily it belongs to.",
        "task_type": "sequence_annotation"
    },
    {
        "system": "You are a drug discovery expert focusing on protein targets.",
        "user_template": "Examine this protein sequence: {sequence}\n\nAssess its potential as a drug target. Discuss druggability, potential binding pockets, and therapeutic relevance.",
        "task_type": "drug_target"
    },
    {
        "system": "You are a protein engineering expert.",
        "user_template": "Looking at this protein sequence: {sequence}\n\nSuggest potential modifications or mutations that could enhance its stability, activity, or introduce new functionalities.",
        "task_type": "protein_engineering"
    },
    {
        "system": "You are an enzymology expert.",
        "user_template": "Analyze this protein sequence: {sequence}\n\nIf this is an enzyme, identify the likely catalytic mechanism, active site residues, and substrate specificity. If not enzymatic, explain its likely function.",
        "task_type": "enzymatic_function"
    },
    {
        "system": "You are a computational biologist specializing in protein properties.",
        "user_template": "Examine this protein sequence: {sequence}\n\nPredict key physicochemical properties such as isoelectric point, hydrophobicity patterns, and potential post-translational modification sites.",
        "task_type": "physicochemical_properties"
    },
    {
        "system": "You are a cellular biology expert.",
        "user_template": "Analyze this protein sequence: {sequence}\n\nPredict its subcellular localization, potential signal peptides or targeting sequences, and its role in cellular processes.",
        "task_type": "cellular_localization"
    },
    {
        "system": "You are an evolutionary biologist studying protein evolution.",
        "user_template": "Looking at this protein sequence: {sequence}\n\nDiscuss its evolutionary conservation, identify functionally important residues, and explain what this tells us about its biological importance.",
        "task_type": "evolutionary_analysis"
    },
    {
        "system": "You are a biochemistry tutor helping students understand protein structure.",
        "user_template": "Describe this protein sequence: {sequence}\n\nExplain its structure and function in simple terms. What makes this protein interesting or important?",
        "task_type": "educational"
    },
    {
        "system": "You are a medical biochemistry expert.",
        "user_template": "Analyze this protein sequence: {sequence}\n\nDiscuss any clinical significance, disease associations, or medical applications related to this protein or similar proteins.",
        "task_type": "clinical_significance"
    },
]


def generate_instruction_prompts(sequence_list: List[str]) -> List[Tuple[str, str]]:
    """
    Generate diverse instruction prompts for proteins.
    
    Args:
        sequence_list: List of protein sequences (amino acid strings)
        
    Returns:
        List of (system_prompt, user_prompt) tuples
    """
    prompts = []
    
    for i, sequence in enumerate(sequence_list):
        # Cycle through templates to ensure diversity
        template = INSTRUCTION_TEMPLATES[i % len(INSTRUCTION_TEMPLATES)]
        
        system_prompt = template["system"]
        # Truncate very long sequences for the prompt to avoid token limits
        sequence_display = sequence[:200] + "..." if len(sequence) > 200 else sequence
        user_prompt = template["user_template"].format(sequence=sequence_display)
        
        prompts.append((system_prompt, user_prompt))
    
    return prompts


def create_sft_dataset(
    protein_parquet_path: str,
    sequence_file_path: str,
    output_parquet_path: str,
    model: str = "gpt-4o-mini",
    use_batch_api: bool = True,
    max_proteins: int = None,
) -> None:
    """
    Generate SFT training dataset for proteins using OpenAI API.
    
    Args:
        protein_parquet_path: Path to protein.parquet file with structural data
        sequence_file_path: Path to text file with protein sequences
        output_parquet_path: Output path for parquet file with messages
        model: OpenAI model to use
        use_batch_api: Whether to use batch API (recommended for cost savings)
        max_proteins: Maximum number of proteins to process (None = all)
    """
    # Load protein structural data
    print(f"Loading protein structures from {protein_parquet_path}...")
    df_structure = pd.read_parquet(protein_parquet_path)
    
    # Load protein sequences
    print(f"Loading sequences from {sequence_file_path}...")
    with open(sequence_file_path, 'r') as f:
        sequence_list = [line.strip() for line in f if line.strip()]
    
    # Validate lengths match
    if len(df_structure) != len(sequence_list):
        print(f"WARNING: Parquet has {len(df_structure)} rows but sequence file has {len(sequence_list)} entries")
        min_len = min(len(df_structure), len(sequence_list))
        df_structure = df_structure.head(min_len)
        sequence_list = sequence_list[:min_len]
        print(f"Using first {min_len} entries")
    
    if max_proteins is not None:
        df_structure = df_structure.head(max_proteins)
        sequence_list = sequence_list[:max_proteins]
    
    num_proteins = len(df_structure)
    print(f"Processing {num_proteins} proteins")
    
    # Generate instruction prompts
    print("Generating instruction prompts...")
    prompts = generate_instruction_prompts(sequence_list)
    
    # Get responses from OpenAI
    print(f"Calling OpenAI API ({model})...")
    if use_batch_api:
        print("Using batch API (this may take a few minutes)...")
        responses = run_batch_requests(
            requests=prompts,
            model=model,
            poll_interval=10.0,  # Poll every 10 seconds
        )
    else:
        from utils.gpt_helper.openai_api import run_sequential_requests
        print("Using sequential API...")
        responses = run_sequential_requests(
            requests=prompts,
            model=model,
        )
    
    # Validate responses
    if len(responses) != num_proteins:
        print(f"WARNING: Expected {num_proteins} responses but got {len(responses)}")
    
    # Create messages column
    print("Creating SFT dataset with messages...")
    messages_list = []
    sequence_clean_list = []
    successful = 0
    
    for idx, (prompt, response, sequence) in enumerate(zip(prompts, responses, sequence_list)):
        if response is None:
            print(f"Warning: No response for protein {idx}, using placeholder...")
            response = "Unable to generate analysis for this protein."
        
        system_prompt, user_prompt = prompt
        
        # Create chat messages following MultiModalSFTDataset format
        # Replace actual sequence with <STRUCTURE> token in user message
        # Handle both truncated and full sequences
        sequence_display = sequence[:200] + "..." if len(sequence) > 200 else sequence
        user_message_clean = user_prompt.replace(sequence_display, "<STRUCTURE>")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_clean},
            {"role": "assistant", "content": response},
        ]
        
        messages_list.append(messages)
        sequence_clean_list.append(sequence)
        successful += 1
    
    # Add new columns to dataframe
    df_output = df_structure.copy()
    df_output['seq'] = sequence_clean_list
    df_output['messages'] = messages_list
    
    # Save to parquet
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_parquet(output_path, index=False)
    
    print(f"\n=== Summary ===")
    print(f"Successfully created {successful}/{num_proteins} examples")
    print(f"Output saved to: {output_parquet_path}")
    print(f"\nDataset columns: {list(df_output.columns)}")
    print(f"  - Structural data: modality, node_feat, pos, edge_index, etc.")
    print(f"  - seq: original protein sequence")
    print(f"  - messages: system, user (with <STRUCTURE> token), assistant")
    print(f"\nUsage with MultiModalSFTDataset:")
    print(f"  - Load parquet and use 'parquet_idx' type in structure field")
    print(f"  - <STRUCTURE> token marks where graph embeddings should be injected")


def main():
    parser = argparse.ArgumentParser(
        description="Generate mock SFT training data for proteins using OpenAI API"
    )
    parser.add_argument(
        '--input-parquet',
        type=str,
        default='data/encoder/test/protein.parquet',
        help='Path to input protein parquet file with structural data'
    )
    parser.add_argument(
        '--input-sequence',
        type=str,
        default='data/encoder/test/raw/protein.txt',
        help='Path to text file with protein sequences (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/protein_sft_train.parquet',
        help='Path to output parquet file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini for cost efficiency)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Use sequential API instead of batch API (faster but more expensive)'
    )
    parser.add_argument(
        '--max-proteins',
        type=int,
        default=None,
        help='Maximum number of proteins to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Verify OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it in your environment or source local_env.sh")
        sys.exit(1)
    
    create_sft_dataset(
        protein_parquet_path=args.input_parquet,
        sequence_file_path=args.input_sequence,
        output_parquet_path=args.output,
        model=args.model,
        use_batch_api=not args.sequential,
        max_proteins=args.max_proteins,
    )


if __name__ == '__main__':
    main()

