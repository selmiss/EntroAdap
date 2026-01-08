import logging
import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from trl import ModelConfig, TrlParser
from tqdm import tqdm

from src.models.inference_configs import ScriptArguments, OctopusConfig
from utils import get_tokenizer
from utils.env_utils import expand_env_vars
from utils.model_utils import load_prepared_octopus_from_checkpoint
from src.data_loader.octopus_collator import (
    MultiModalInferenceCollator,
    preprocess_inference_dataset,
)
from types import SimpleNamespace


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_inference_data(
    input_file: str,
    tokenizer,
    max_seq_length: int = 2048,
    structure_tokens: List[str] = None,
) -> DatasetDict:
    """
    Load and preprocess data for inference.
    
    Supports:
    - Combined parquet format (with messages + graph columns)
    - JSONL format (text only, or with structure metadata)
    
    Args:
        input_file: Path to input file (.parquet or .jsonl)
        tokenizer: Tokenizer for text processing
        max_seq_length: Maximum sequence length
        structure_tokens: List of structure tokens to recognize
    
    Returns:
        Preprocessed dataset ready for inference
    """
    input_path = Path(input_file)
    
    if input_path.suffix == '.parquet':
        # Load parquet directly (supports combined format with graph columns)
        logger.info(f"Loading parquet file: {input_file}")
        dataset = load_dataset('parquet', data_files=input_file, split='train')
        dataset_dict = DatasetDict({'test': dataset})
    elif input_path.suffix in ['.json', '.jsonl']:
        # Load JSONL
        logger.info(f"Loading JSONL file: {input_file}")
        dataset = load_dataset('json', data_files=input_file, split='train')
        dataset_dict = DatasetDict({'test': dataset})
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .parquet or .jsonl")
    
    logger.info(f"Loaded {len(dataset_dict['test'])} samples")
    
    # Preprocess for inference (tokenize prompts, find structure tokens, etc.)
    processed = preprocess_inference_dataset(
        dataset_dict,
        tokenizer,
        split='test',
        max_seq_length=max_seq_length,
        structure_tokens=structure_tokens,
    )
    
    return processed


def run_inference(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: str = 'cuda',
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Run inference on a dataloader and collect results.
    
    Args:
        model: Octopus model
        dataloader: DataLoader with inference data
        tokenizer: Tokenizer for decoding
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
    
    Returns:
        List of results with predictions and references
    """
    model.eval()
    results = []
    
    logger.info(f"Generation parameters: max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Move graph data if present
            graph_node_nums = []
            if 'graph_data' in batch:
                graph_data = batch['graph_data']
                graph_data['value']['node_feat'] = graph_data['value']['node_feat'].to(device)
                graph_data['value']['pos'] = graph_data['value']['pos'].to(device)
                if 'edge_index' in graph_data['value']:
                    graph_data['value']['edge_index'] = graph_data['value']['edge_index'].to(device)
                if 'edge_attr' in graph_data['value']:
                    graph_data['value']['edge_attr'] = graph_data['value']['edge_attr'].to(device)
                if 'edge_feat_dist' in graph_data['value']:
                    graph_data['value']['edge_feat_dist'] = graph_data['value']['edge_feat_dist'].to(device)
                if 'chem_edge_index' in graph_data['value']:
                    graph_data['value']['chem_edge_index'] = graph_data['value']['chem_edge_index'].to(device)
                if 'chem_edge_feat_cat' in graph_data['value']:
                    graph_data['value']['chem_edge_feat_cat'] = graph_data['value']['chem_edge_feat_cat'].to(device)
                batch['graph_data'] = graph_data
            
            if 'batch' in batch:
                batch['batch'] = batch['batch'].to(device)
                # Calculate number of nodes per graph in the batch
                batch_tensor = batch['batch']
                for graph_idx in range(batch_tensor.max().item() + 1):
                    num_nodes = (batch_tensor == graph_idx).sum().item()
                    graph_node_nums.append(num_nodes)
            
            if 'instr_positions' in batch:
                batch['instr_positions'] = batch['instr_positions'].to(device)
            
            if 'patch_positions' in batch:
                batch['patch_positions'] = batch['patch_positions'].to(device)
            
            # Generate
            outputs, num_injected_patches = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_data=batch.get('graph_data'),
                batch=batch.get('batch'),
                instr_positions=batch.get('instr_positions'),
                patch_positions=batch.get('patch_positions'),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                return_patch_tokens_count=True,
            )
            
            # Decode prompts for reference
            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            # input_length = input_ids.shape[1] + num_injected_patches
        
            generated_tokens = outputs
            
            predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # Collect results
            for i, pred in enumerate(predictions):
                result = {
                    'sample_id': batch.get('sample_id', [f'sample_{i}'])[i],
                    'prompt': prompts[i],
                    'prediction': pred,
                    'reference': batch.get('reference_text', [''])[i],
                    'injected_tokens_number': num_injected_patches[i] if isinstance(num_injected_patches, list) else num_injected_patches,
                    'graph_node_num': graph_node_nums[i] if i < len(graph_node_nums) else 0,
                }
                results.append(result)
    
    return results


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save inference results to file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved {len(results)} results to {output_file}")


def main(script_args, model_config, octopus_config):
    """Main inference pipeline."""
    
    combined_config = SimpleNamespace(**vars(model_config), **vars(octopus_config))
    model_config.model_name_or_path = octopus_config.octopus_checkpoint_path

    # Load model
    logger.info(f"Loading model from {octopus_config.octopus_checkpoint_path}")
    model = load_prepared_octopus_from_checkpoint(octopus_config.octopus_checkpoint_path, combined_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, 
        revision=model_config.model_revision, 
        trust_remote_code=model_config.trust_remote_code, 
        fix_mistral_regex=True
    )
    
    # Only override chat template if explicitly provided
    # Otherwise, use the template saved with the checkpoint
    if script_args.chat_template is not None:
        logger.info("Using custom chat template from config")
        tokenizer.chat_template = script_args.chat_template
    else:
        logger.info(f"Using chat template from checkpoint: {tokenizer.chat_template[:100] if tokenizer.chat_template else 'None'}...")
    
    # Verify we have a template
    if tokenizer.chat_template is None:
        raise ValueError(
            "No chat template found! Either:\n"
            "1. Checkpoint should have a saved template, or\n"
            "2. Provide --chat_template in config"
        )
    
    # Load and preprocess data
    logger.info(f"Loading data from {script_args.input_file}")
    structure_tokens = ["<STRUCTURE>", "<mol>", "<STRUCT>", "<DNA>"]  # Can be extended to support custom tokens
    dataset = load_inference_data(
        script_args.input_file,
        tokenizer,
        max_seq_length=octopus_config.max_seq_length,
        structure_tokens=structure_tokens,
    )
    
    # Create dataloader
    collator = MultiModalInferenceCollator(
        tokenizer=tokenizer,
        max_length=octopus_config.max_seq_length,
        structure_tokens=structure_tokens,
    )
    
    dataloader = DataLoader(
        dataset['test'],
        batch_size=script_args.batch_size,
        collate_fn=collator,
        shuffle=False,  # Keep order for evaluation
    )
    
    logger.info(f"Created dataloader with {len(dataloader)} batches")
    
    # Run inference
    results = run_inference(
        model,
        dataloader,
        tokenizer,
        device=device,
        max_new_tokens=script_args.max_new_tokens,
        temperature=script_args.temperature,
        top_p=script_args.top_p,
    )
    
    # Save results
    save_results(results, script_args.output_dir)
    


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, ModelConfig, OctopusConfig))
    script_args, model_config, octopus_config = parser.parse_args_and_config()
    expand_env_vars(script_args, model_config, octopus_config)
    main(script_args, model_config, octopus_config)

