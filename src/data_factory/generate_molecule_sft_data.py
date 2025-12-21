"""
Generate mock SFT training data for molecules using OpenAI Batch API.

This script:
1. Reads molecule.parquet containing structural information
2. Reads corresponding SMILES strings from raw file
3. Generates diverse instruction-response pairs using OpenAI batch API
4. Outputs parquet format with messages column for SFT training

Output parquet schema:
- modality: str
- node_feat: array
- pos: array
- edge_index: array
- chem_edge_index: array
- chem_edge_feat_cat: array
- edge_feat_dist: array
- smiles: str (original SMILES)
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
        "system": "You are a chemistry expert assistant that analyzes molecular structures and properties.",
        "user_template": "Analyze this molecular structure with SMILES: {smiles}\n\nDescribe its key chemical properties, functional groups, and potential applications. Be concise but informative.",
        "task_type": "property_analysis"
    },
    {
        "system": "You are a medicinal chemistry expert specializing in drug discovery.",
        "user_template": "Examine this molecule (SMILES: {smiles}) and discuss its potential as a drug candidate. Consider factors like drug-likeness, bioavailability, and potential biological activity.",
        "task_type": "drug_discovery"
    },
    {
        "system": "You are an organic chemistry expert.",
        "user_template": "Given this molecular structure (SMILES: {smiles}), identify and describe all functional groups present and explain their chemical significance.",
        "task_type": "functional_groups"
    },
    {
        "system": "You are a computational chemist.",
        "user_template": "Analyze this molecule (SMILES: {smiles}) and predict its key physical and chemical properties (e.g., polarity, solubility, reactivity). Explain your reasoning.",
        "task_type": "property_prediction"
    },
    {
        "system": "You are a chemistry tutor helping students understand molecular structures.",
        "user_template": "Describe this molecular structure (SMILES: {smiles}) in simple terms. What makes this molecule interesting or important?",
        "task_type": "educational"
    },
    {
        "system": "You are a toxicology expert.",
        "user_template": "Examine this molecule (SMILES: {smiles}) and assess potential toxicity concerns or safety considerations based on its structural features.",
        "task_type": "toxicology"
    },
    {
        "system": "You are a synthetic chemistry expert.",
        "user_template": "Looking at this molecule (SMILES: {smiles}), suggest potential synthetic routes or key reactions that could be used to synthesize this compound.",
        "task_type": "synthesis"
    },
    {
        "system": "You are a molecular biology expert.",
        "user_template": "Analyze this molecule (SMILES: {smiles}) and discuss how it might interact with biological systems, including potential targets or mechanisms of action.",
        "task_type": "biological_interaction"
    },
]


def generate_instruction_prompts(smiles_list: List[str]) -> List[Tuple[str, str]]:
    """
    Generate diverse instruction prompts for molecules.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        List of (system_prompt, user_prompt) tuples
    """
    prompts = []
    
    for i, smiles in enumerate(smiles_list):
        # Cycle through templates to ensure diversity
        template = INSTRUCTION_TEMPLATES[i % len(INSTRUCTION_TEMPLATES)]
        
        system_prompt = template["system"]
        user_prompt = template["user_template"].format(smiles=smiles)
        
        prompts.append((system_prompt, user_prompt))
    
    return prompts


def create_sft_dataset(
    molecule_parquet_path: str,
    smiles_file_path: str,
    output_parquet_path: str,
    model: str = "gpt-4o-mini",
    use_batch_api: bool = True,
    max_molecules: int = None,
) -> None:
    """
    Generate SFT training dataset for molecules using OpenAI API.
    
    Args:
        molecule_parquet_path: Path to molecule.parquet file with structural data
        smiles_file_path: Path to text file with SMILES strings
        output_parquet_path: Output path for parquet file with messages
        model: OpenAI model to use
        use_batch_api: Whether to use batch API (recommended for cost savings)
        max_molecules: Maximum number of molecules to process (None = all)
    """
    # Load molecule structural data
    print(f"Loading molecule structures from {molecule_parquet_path}...")
    df_structure = pd.read_parquet(molecule_parquet_path)
    
    # Load SMILES strings
    print(f"Loading SMILES from {smiles_file_path}...")
    with open(smiles_file_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    # Validate lengths match
    if len(df_structure) != len(smiles_list):
        print(f"WARNING: Parquet has {len(df_structure)} rows but SMILES file has {len(smiles_list)} entries")
        min_len = min(len(df_structure), len(smiles_list))
        df_structure = df_structure.head(min_len)
        smiles_list = smiles_list[:min_len]
        print(f"Using first {min_len} entries")
    
    if max_molecules is not None:
        df_structure = df_structure.head(max_molecules)
        smiles_list = smiles_list[:max_molecules]
    
    num_molecules = len(df_structure)
    print(f"Processing {num_molecules} molecules")
    
    # Generate instruction prompts
    print("Generating instruction prompts...")
    prompts = generate_instruction_prompts(smiles_list)
    
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
    if len(responses) != num_molecules:
        print(f"WARNING: Expected {num_molecules} responses but got {len(responses)}")
    
    # Create messages column
    print("Creating SFT dataset with messages...")
    messages_list = []
    smiles_clean_list = []
    successful = 0
    
    for idx, (prompt, response, smiles) in enumerate(zip(prompts, responses, smiles_list)):
        if response is None:
            print(f"Warning: No response for molecule {idx}, using placeholder...")
            response = "Unable to generate analysis for this molecule."
        
        system_prompt, user_prompt = prompt
        
        # Create chat messages following MultiModalSFTDataset format
        # Replace actual SMILES with <STRUCTURE> token in user message
        user_message_clean = user_prompt.replace(f"SMILES: {smiles}", "<STRUCTURE>")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message_clean},
            {"role": "assistant", "content": response},
        ]
        
        messages_list.append(messages)
        smiles_clean_list.append(smiles)
        successful += 1
    
    # Add new columns to dataframe
    df_output = df_structure.copy()
    df_output['smiles'] = smiles_clean_list
    df_output['messages'] = messages_list
    
    # Save to parquet
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_parquet(output_path, index=False)
    
    print(f"\n=== Summary ===")
    print(f"Successfully created {successful}/{num_molecules} examples")
    print(f"Output saved to: {output_parquet_path}")
    print(f"\nDataset columns: {list(df_output.columns)}")
    print(f"  - Structural data: modality, node_feat, pos, edge_index, etc.")
    print(f"  - SMILES: original SMILES string")
    print(f"  - messages: system, user (with <STRUCTURE> token), assistant")
    print(f"\nUsage with MultiModalSFTDataset:")
    print(f"  - Load parquet and use 'parquet_idx' type in structure field")
    print(f"  - <STRUCTURE> token marks where graph embeddings should be injected")


def main():
    parser = argparse.ArgumentParser(
        description="Generate mock SFT training data for molecules using OpenAI API"
    )
    parser.add_argument(
        '--input-parquet',
        type=str,
        default='data/encoder/test/molecule.parquet',
        help='Path to input molecule parquet file with structural data'
    )
    parser.add_argument(
        '--input-smiles',
        type=str,
        default='data/encoder/test/raw/molecule.txt',
        help='Path to text file with SMILES strings (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/molecule_sft_train.parquet',
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
        '--max-molecules',
        type=int,
        default=None,
        help='Maximum number of molecules to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Verify OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it in your environment or source local_env.sh")
        sys.exit(1)
    
    create_sft_dataset(
        molecule_parquet_path=args.input_parquet,
        smiles_file_path=args.input_smiles,
        output_parquet_path=args.output,
        model=args.model,
        use_batch_api=not args.sequential,
        max_molecules=args.max_molecules,
    )


if __name__ == '__main__':
    main()
