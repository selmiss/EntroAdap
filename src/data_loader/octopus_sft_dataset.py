"""
Multi-modal SFT Dataset and Collator for Graph + Text Training

This module integrates graph structures (protein/molecule) with text instructions
for supervised fine-tuning of Octopus.

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

from .aa_dataset import GraphDataset
from src.data_factory.protein.pdbid_to_feature import pdbid_to_features
from src.data_factory.molecule.mol_structure import generate_2d_3d_from_smiles


class MultiModalSFTDataset(Dataset):
    """
    Dataset combining graph structures with text instructions for SFT.
    
    NOTE: This class returns graph data in NESTED format (graph_data dict).
    The current training pipeline uses FLAT format (direct columns: modality, node_feat, pos, etc.)
    and loads data via HuggingFace datasets directly. This class is primarily used for
    data generation and testing.
    
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
        max_atoms: Optional maximum number of atoms per structure. Structures exceeding
            this limit will be skipped during loading. None means no limit.
        max_edges: Optional maximum number of edges per structure. Structures exceeding
            this limit will be skipped during loading. None means no limit.
        skip_on_error: If True, skip samples that fail to load or exceed thresholds.
            If False, raise exceptions. Default: True.
    """
    
    def __init__(
        self,
        dataset_path: str,
        graph_parquet_path: Optional[str] = None,
        structure_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_combined_parquet: bool = False,
        max_atoms: Optional[int] = None,
        max_edges: Optional[int] = None,
        skip_on_error: bool = True,
    ):
        self.use_combined_parquet = use_combined_parquet
        self.structure_dir = structure_dir
        
        # Store runtime filtering parameters
        self.max_atoms = max_atoms
        self.max_edges = max_edges
        self.skip_on_error = skip_on_error
        self._filtered_count = 0
        self._error_count = 0
        
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
                self.graph_dataset = GraphDataset(
                    graph_parquet_path, 
                    cache_dir=cache_dir,
                    max_atoms=max_atoms,
                    max_edges=max_edges,
                    skip_on_error=skip_on_error,
                )
    
    def __len__(self) -> int:
        if self.use_combined_parquet:
            return len(self.df)
        else:
            return len(self.text_dataset)
    
    def get_filter_stats(self) -> Dict[str, int]:
        """Get statistics about filtered samples."""
        return {
            'filtered_count': self._filtered_count,
            'error_count': self._error_count,
        }
    
    def _load_graph_from_parquet_row(self, row) -> Optional[Dict[str, Any]]:
        """Load graph structure from a combined parquet row with optional filtering."""
        # Early size check before tensor conversion
        if self.max_atoms is not None or self.max_edges is not None:
            # Check atom count
            if hasattr(row['node_feat'], '__len__'):
                num_atoms = len(row['node_feat'])
            else:
                node_feat_arr = np.array(row['node_feat'])
                num_atoms = len(node_feat_arr) if node_feat_arr.ndim > 0 else 0
            
            # Check edge count
            num_edges = 0
            if 'edge_index' in row and row['edge_index'] is not None:
                edge_index_data = row['edge_index']
                if hasattr(edge_index_data, '__len__'):
                    edge_index_arr = np.array(edge_index_data)
                    if edge_index_arr.ndim >= 2:
                        num_edges = edge_index_arr.shape[1] if edge_index_arr.shape[0] > 0 else 0
            
            # For molecules, also count chemical edges
            modality = row.get('modality', '')
            if modality == 'molecule' and 'chem_edge_index' in row and row['chem_edge_index'] is not None:
                chem_edge_data = row['chem_edge_index']
                if hasattr(chem_edge_data, '__len__'):
                    chem_edge_arr = np.array(chem_edge_data)
                    if chem_edge_arr.ndim >= 2:
                        num_edges += chem_edge_arr.shape[1] if chem_edge_arr.shape[0] > 0 else 0
            
            # Apply thresholds
            if self.max_atoms is not None and num_atoms > self.max_atoms:
                self._filtered_count += 1
                return None
            
            if self.max_edges is not None and num_edges > self.max_edges:
                self._filtered_count += 1
                return None
        
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
                - graph_data: Optional graph structure (may be None if filtered)
                - structure_token: Token indicating where structure should be injected
        
        Note:
            When skip_on_error=True and a sample is filtered, this will recursively
            try the next sample. Graph data may be None if structure exceeds thresholds.
        """
        try:
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
                
                # Load graph structure from the same row (may return None if filtered)
                graph_data = self._load_graph_from_parquet_row(row)
                if graph_data is not None:
                    result['graph_data'] = graph_data
                elif self.skip_on_error and (self.max_atoms is not None or self.max_edges is not None):
                    # Structure was filtered - try next sample
                    next_idx = (idx + 1) % len(self.df)
                    if next_idx == idx:
                        # Went full circle - just return without graph_data
                        return result
                    return self.__getitem__(next_idx)
                
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
        
        except Exception as e:
            self._error_count += 1
            if self.skip_on_error:
                # Try next sample (with wraparound)
                next_idx = (idx + 1) % len(self)
                if next_idx == idx:
                    raise RuntimeError(f"All samples failed to load") from e
                # Only log first few errors to avoid spam
                if self._error_count <= 5:
                    print(f"Warning: Error loading sample {idx}: {e}. Skipping to next sample.")
                return self.__getitem__(next_idx)
            else:
                raise
