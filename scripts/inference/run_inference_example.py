#!/usr/bin/env python3
"""
Example script demonstrating programmatic usage of the inference module.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.runner.inference import (
    load_octopus_model,
    load_input_data,
    run_inference,
    InferenceArguments,
)


def main():
    """Example of programmatic inference."""
    
    # Define arguments
    args = InferenceArguments(
        checkpoint_path="./checkpoints/octopus/Qwen2-0.5B",
        input_file="data/sft/protein/protein_sft_flat.parquet",
        output_file="results/inference_output.jsonl",
        max_samples=10,  # Process only 10 samples for quick test
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        device="cuda",  # or "cpu"
    )
    
    print(f"Loading model from {args.checkpoint_path}...")
    model, tokenizer = load_octopus_model(args.checkpoint_path, args.device)
    
    print(f"Loading data from {args.input_file}...")
    df = load_input_data(args.input_file, args.max_samples)
    
    print(f"Running inference on {len(df)} samples...")
    results = []
    
    for idx, row in df.iterrows():
        sample = row.to_dict()
        
        try:
            result = run_inference(model, tokenizer, sample, args)
            result["sample_id"] = idx
            results.append(result)
            
            # Print sample result
            print(f"\n{'='*80}")
            print(f"Sample {idx}")
            print(f"Modality: {result['modality']}")
            print(f"Input prompt (first 200 chars): {result['input_prompt'][:200]}...")
            print(f"Generated: {result['generated_text']}")
            if 'label' in result:
                print(f"Label: {result['label']}")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            continue
    
    # Save results
    import json
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nResults saved to {args.output_file}")
    print(f"Processed {len(results)} samples successfully")


if __name__ == "__main__":
    main()

