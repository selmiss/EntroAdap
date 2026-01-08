#!/usr/bin/env python3
"""
Process text-only instruction datasets into SFT format.

This script processes JSON datasets with instruction/input/output format
into the SFT parquet format with messages field (no graph data needed).

Input format:
{
    "instruction": "...",
    "input": "...",  # prompt/question
    "output": "...",  # answer
    "metadata": {
        "split": "train" or "test"
    }
}

Output format (parquet):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm


def load_json_data(json_path: Path) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    print(f"Loading data from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        # If it's a dict, it might have a 'data' key or similar
        if 'data' in data:
            data = data['data']
        elif 'examples' in data:
            data = data['examples']
        else:
            # Assume it's a single example wrapped in a dict
            data = [data]
    
    print(f"Loaded {len(data)} examples")
    return data


def convert_to_messages(
    example: Dict[str, Any],
    system_prompt: str = None
) -> Dict[str, Any]:
    """
    Convert a single example to messages format.
    
    Args:
        example: Dict with 'instruction', 'input', 'output' keys
        system_prompt: Optional system prompt to use
    
    Returns:
        Dict with 'messages' field
    """
    instruction = example.get('instruction', '').strip()
    user_input = example.get('input', '').strip()
    output = example.get('output', '').strip()
    
    # Construct user message
    if instruction and user_input:
        # Both instruction and input present
        user_content = f"{instruction}\n\n{user_input}"
    elif instruction:
        # Only instruction
        user_content = instruction
    elif user_input:
        # Only input
        user_content = user_input
    else:
        raise ValueError("Example must have either 'instruction' or 'input'")
    
    # Build messages
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add user message
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    # Add assistant response
    messages.append({
        "role": "assistant",
        "content": output
    })
    
    return {"messages": messages}


def process_dataset(
    json_path: Path,
    output_dir: Path,
    system_prompt: str = None,
    split_by_metadata: bool = True
):
    """
    Process JSON dataset into SFT parquet format.
    
    Args:
        json_path: Path to input JSON file
        output_dir: Directory to save output parquet files
        system_prompt: Optional system prompt to add to all examples
        split_by_metadata: If True, split into train/test based on metadata
    """
    # Load data
    data = load_json_data(json_path)
    
    # Convert to messages format
    print("Converting to messages format...")
    converted_data = []
    
    for example in tqdm(data):
        try:
            converted = convert_to_messages(example, system_prompt)
            
            # Add split information if available
            if 'metadata' in example and 'split' in example['metadata']:
                converted['split'] = example['metadata']['split']
            else:
                converted['split'] = 'train'  # Default to train
            
            converted_data.append(converted)
        except Exception as e:
            print(f"Warning: Failed to convert example: {e}")
            continue
    
    print(f"Successfully converted {len(converted_data)} examples")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split by train/test if requested
    if split_by_metadata:
        # Group by split
        splits = {}
        for item in converted_data:
            split = item.get('split', 'train')
            if split not in splits:
                splits[split] = []
            splits[split].append(item)
        
        # Save each split
        for split_name, split_data in splits.items():
            output_path = output_dir / f"{split_name}.parquet"
            df = pd.DataFrame(split_data)
            df.to_parquet(output_path, index=False)
            print(f"Saved {len(split_data)} examples to {output_path}")
    else:
        # Save all data together
        output_path = output_dir / "data.parquet"
        df = pd.DataFrame(converted_data)
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(converted_data)} examples to {output_path}")
    
    print("Processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Process text-only instruction datasets into SFT format"
    )
    parser.add_argument(
        'input_json',
        type=str,
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/benchmark',
        help='Output directory for parquet files (default: data/benchmark)'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default=None,
        help='System prompt to add to all examples'
    )
    parser.add_argument(
        '--no_split',
        action='store_true',
        help='Do not split by metadata, save all data together'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        help='Name for the dataset (used as subdirectory name)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output directory
    output_dir = Path(args.output_dir)
    if args.dataset_name:
        output_dir = output_dir / args.dataset_name
    else:
        # Use input filename (without extension) as subdirectory
        dataset_name = input_path.stem
        output_dir = output_dir / dataset_name
    
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"System prompt: {args.system_prompt or 'None'}")
    print(f"Split by metadata: {not args.no_split}")
    print()
    
    # Process dataset
    process_dataset(
        json_path=input_path,
        output_dir=output_dir,
        system_prompt=args.system_prompt,
        split_by_metadata=not args.no_split
    )


if __name__ == '__main__':
    main()

