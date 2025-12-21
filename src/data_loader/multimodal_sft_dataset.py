"""
Multi-modal SFT Dataset and Collator for Graph + Text Training

This module integrates graph structures (protein/molecule) with text instructions
for supervised fine-tuning of MultiModalLLM.

Dataset Format (JSONL):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "Analyze this protein structure: <STRUCTURE>"},
        {"role": "assistant", "content": "..."}
    ],
    "structure": {
        "type": "parquet_idx",  // or "pdb_id", "smiles", "inline"
        "value": 123,  // parquet index, PDB ID, SMILES string, or inline data
        "modality": "protein"  // or "molecule"
    }
}

OR Combined Parquet Format:
Single parquet file with columns: modality, node_feat, pos, edge_index, messages, etc.
Each row contains both structural data and instruction messages.

The <STRUCTURE> token in messages indicates where the graph should be injected.
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import pandas as pd
import numpy as np

from .graph_dataset import GraphDataset
from src.data_factory.protein.pdbid_to_feature import pdbid_to_features
from src.data_factory.molecule.mol_structure import generate_2d_3d_from_smiles


class MultiModalSFTDataset(Dataset):
    """
    Dataset combining graph structures with text instructions for SFT.
    
    Supports two input formats:
    1. JSONL + separate parquet (original): 
       - dataset_path: JSONL with messages and structure metadata
       - graph_parquet_path: Separate parquet with structural data
    
    2. Combined parquet (new):
       - dataset_path: Single parquet with both messages and structural data
       - Set use_combined_parquet=True
    
    Args:
        dataset_path: Path to JSONL or combined parquet file
        graph_parquet_path: Optional path to separate parquet with graphs (JSONL mode only)
        structure_dir: Directory containing PDB structures (for on-the-fly loading)
        cache_dir: Cache directory for HF datasets
        use_combined_parquet: If True, load from combined parquet format
    """
    
    def __init__(
        self,
        dataset_path: str,
        graph_parquet_path: Optional[str] = None,
        structure_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_combined_parquet: bool = False,
    ):
        self.use_combined_parquet = use_combined_parquet
        self.structure_dir = structure_dir
        
        if use_combined_parquet:
            # Load combined parquet format
            self.df = pd.read_parquet(dataset_path)
            self.text_dataset = None
            self.graph_dataset = None
            print(f"Loaded {len(self.df)} examples from combined parquet: {dataset_path}")
        else:
            # Load JSONL format with separate graph parquet
            self.text_dataset = load_dataset(
                'json',
                data_files=dataset_path,
                cache_dir=cache_dir,
                split='train',
            )
            self.df = None
            
            # Load graph dataset if provided
            self.graph_dataset = None
            if graph_parquet_path is not None:
                self.graph_dataset = GraphDataset(graph_parquet_path, cache_dir=cache_dir)
    
    def __len__(self) -> int:
        if self.use_combined_parquet:
            return len(self.df)
        else:
            return len(self.text_dataset)
    
    def _load_graph_from_parquet_row(self, row) -> Dict[str, Any]:
        """Load graph structure from a combined parquet row."""
        # Convert nested arrays properly
        node_feat_arr = np.array(row['node_feat'])
        if node_feat_arr.dtype == object:
            node_feat_arr = np.vstack([np.array(x) for x in node_feat_arr])
        
        pos_arr = np.array(row['pos'])
        if pos_arr.dtype == object:
            pos_arr = np.vstack([np.array(x) for x in pos_arr])
        
        edge_index_arr = np.array(row['edge_index'])
        if edge_index_arr.dtype == object:
            edge_index_arr = np.vstack([np.array(x) for x in edge_index_arr])
        
        graph_data = {
            'modality': row['modality'],
            'value': {
                'node_feat': torch.tensor(node_feat_arr, dtype=torch.float32),
                'pos': torch.tensor(pos_arr, dtype=torch.float32),
                'edge_index': torch.tensor(edge_index_arr, dtype=torch.long),
            }
        }
        
        # Add optional fields
        if 'chem_edge_index' in row and row['chem_edge_index'] is not None:
            chem_edge_index_arr = np.array(row['chem_edge_index'])
            if chem_edge_index_arr.dtype == object:
                chem_edge_index_arr = np.vstack([np.array(x) for x in chem_edge_index_arr])
            graph_data['value']['chem_edge_index'] = torch.tensor(
                chem_edge_index_arr, dtype=torch.long
            )
        
        if 'chem_edge_feat_cat' in row and row['chem_edge_feat_cat'] is not None:
            chem_edge_feat_arr = np.array(row['chem_edge_feat_cat'])
            if chem_edge_feat_arr.dtype == object:
                chem_edge_feat_arr = np.vstack([np.array(x) for x in chem_edge_feat_arr])
            graph_data['value']['chem_edge_feat_cat'] = torch.tensor(
                chem_edge_feat_arr, dtype=torch.long
            )
        
        if 'edge_feat_dist' in row and row['edge_feat_dist'] is not None:
            edge_feat_dist_arr = np.array(row['edge_feat_dist'])
            if edge_feat_dist_arr.dtype == object:
                edge_feat_dist_arr = np.vstack([np.array(x) for x in edge_feat_dist_arr])
            graph_data['value']['edge_feat_dist'] = torch.tensor(
                edge_feat_dist_arr, dtype=torch.float32
            )
        
        if 'edge_attr' in row and row['edge_attr'] is not None:
            edge_attr_arr = np.array(row['edge_attr'])
            if edge_attr_arr.dtype == object:
                edge_attr_arr = np.vstack([np.array(x) for x in edge_attr_arr])
            graph_data['value']['edge_attr'] = torch.tensor(
                edge_attr_arr, dtype=torch.float32
            )
        
        return graph_data
    
    def _load_graph_from_source(self, structure_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load graph structure from various sources."""
        source_type = structure_info.get('type')
        value = structure_info.get('value')
        modality = structure_info.get('modality', 'protein')
        
        if source_type == 'parquet_idx' and self.graph_dataset is not None:
            # Load from pre-processed parquet
            return self.graph_dataset[value]
        
        elif source_type == 'pdb_id' and self.structure_dir is not None:
            # Load protein from PDB on-the-fly
            result = pdbid_to_features(
                pdb_id=value,
                structure_dir=self.structure_dir,
                graph_radius=6.0,
                ca_only=False,
            )
            if result is None:
                return None
            
            return {
                'modality': 'protein',
                'value': {
                    'node_feat': torch.tensor(result['node_feat'], dtype=torch.float32),
                    'pos': torch.tensor(result['coordinates'], dtype=torch.float32),
                    'edge_index': torch.tensor(result['edge_index'], dtype=torch.long),
                    'edge_attr': torch.tensor(result['edge_attr'], dtype=torch.float32).unsqueeze(-1),
                }
            }
        
        elif source_type == 'smiles':
            # Load molecule from SMILES on-the-fly
            atoms, graph_2d, coords_3d = generate_2d_3d_from_smiles(value)
            if atoms is None or graph_2d is None or coords_3d is None:
                return None
            
            graph_data = {
                'modality': 'molecule',
                'value': {
                    'node_feat': graph_2d['node_feat'].float(),
                    'pos': torch.tensor(coords_3d, dtype=torch.float32),
                    'edge_index': graph_2d['edge_index'],
                }
            }
            
            # Add optional fields
            if 'edge_feat_dist' in graph_2d:
                graph_data['value']['edge_feat_dist'] = graph_2d['edge_feat_dist']
            if 'chem_edge_index' in graph_2d:
                graph_data['value']['chem_edge_index'] = graph_2d['chem_edge_index']
            if 'chem_edge_feat_cat' in graph_2d:
                graph_data['value']['chem_edge_feat_cat'] = graph_2d['chem_edge_feat_cat']
            
            return graph_data
        
        elif source_type == 'inline':
            # Graph data embedded directly in the example
            # Need to convert lists to tensors if not already
            graph_data = value
            if isinstance(graph_data, dict) and 'value' in graph_data:
                value_dict = graph_data['value']
                # Convert lists to tensors
                if not isinstance(value_dict.get('node_feat'), torch.Tensor):
                    value_dict['node_feat'] = torch.tensor(value_dict['node_feat'], dtype=torch.float32)
                if not isinstance(value_dict.get('pos'), torch.Tensor):
                    value_dict['pos'] = torch.tensor(value_dict['pos'], dtype=torch.float32)
                if not isinstance(value_dict.get('edge_index'), torch.Tensor):
                    value_dict['edge_index'] = torch.tensor(value_dict['edge_index'], dtype=torch.long)
                if 'edge_attr' in value_dict and not isinstance(value_dict['edge_attr'], torch.Tensor):
                    value_dict['edge_attr'] = torch.tensor(value_dict['edge_attr'], dtype=torch.float32)
            return graph_data
        
        return None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single example with text and optional graph structure.
        
        Returns:
            Dictionary with:
                - messages: List of chat messages
                - graph_data: Optional graph structure
                - structure_token: Token indicating where structure should be injected
        """
        if self.use_combined_parquet:
            # Load from combined parquet format
            row = self.df.iloc[idx]
            
            # Extract messages
            messages = row['messages']
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            
            result = {
                'messages': messages,
                'structure_token': '<STRUCTURE>',
            }
            
            # Load graph structure from the same row
            graph_data = self._load_graph_from_parquet_row(row)
            if graph_data is not None:
                result['graph_data'] = graph_data
            
            # Add SMILES for reference if available
            if 'smiles' in row:
                result['smiles'] = row['smiles']
            
            return result
        
        else:
            # Original JSONL format
            item = self.text_dataset[idx]
            
            result = {
                'messages': item['messages'],
                'structure_token': '<STRUCTURE>',  # Default token
            }
            
            # Load graph if structure info is provided
            if 'structure' in item and item['structure'] is not None:
                graph_data = self._load_graph_from_source(item['structure'])
                if graph_data is not None:
                    result['graph_data'] = graph_data
            
            # Allow custom structure token
            if 'structure_token' in item:
                result['structure_token'] = item['structure_token']
            
            return result


def preprocess_multimodal_sft_dataset(
    dataset: Union[Dataset, Any],
    tokenizer,
    split: str = 'train',
    max_seq_length: int = 1024,
) -> Dataset:
    """
    Preprocess multimodal SFT dataset by tokenizing messages.
    
    This function:
    1. Applies chat template to messages
    2. Tokenizes text
    3. Identifies instruction positions (user messages)
    4. Preserves graph_data for collator
    
    Args:
        dataset: MultiModalSFTDataset or HF Dataset
        tokenizer: Tokenizer with chat template
        split: Split name (unused, for compatibility)
        max_seq_length: Maximum sequence length
    
    Returns:
        Processed dataset with tokenized inputs
    """
    def _preprocess_example(example):
        """Process a single example."""
        messages = example['messages']
        structure_token = example.get('structure_token', '<STRUCTURE>')
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # Padding handled by collator
        )
        
        # Create labels (copy of input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'][:]
        
        # Find instruction positions (user message tokens)
        # Simple heuristic: find user message content in tokenized sequence
        instr_positions = []
        for msg in messages:
            if msg['role'] == 'user':
                # Tokenize just the user content to find its positions
                user_text = msg['content']
                user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
                
                # Find where these tokens appear in the full sequence
                # (Simple substring matching - more robust methods possible)
                input_ids = tokenized['input_ids']
                for i in range(len(input_ids) - len(user_tokens) + 1):
                    if input_ids[i:i+len(user_tokens)] == user_tokens:
                        # Found user message, store some positions
                        # Take first few tokens as instruction representation
                        instr_positions.extend(list(range(i, min(i+10, len(input_ids)))))
                        break
        
        # Store instruction positions (will be padded by collator)
        tokenized['instr_positions'] = instr_positions if instr_positions else [0]
        
        # Preserve graph data if present
        if 'graph_data' in example:
            tokenized['graph_data'] = example['graph_data']
        
        return tokenized
    
    # Process dataset
    if hasattr(dataset, 'map'):
        processed = dataset.map(
            _preprocess_example,
            desc="Tokenizing multimodal dataset",
        )
    else:
        # If it's a list or single example
        processed = [_preprocess_example(ex) for ex in dataset]
    
    return processed

