"""
Geometric Encoder

Combines FeatureEmbedder and EGNN Backbone into a unified encoder for processing
graphs with 3D coordinates.

Key design features:
- Modality-agnostic EGNN processing (no 'protein' or 'molecule' logic here)
- All modality-specific handling is done in FeatureEmbedder
- All embeddings are mapped to consistent hidden_dim through linear projections
- FeatureEmbedder handles edge concatenation for molecules internally
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any

from .components.feature_embedder import FeatureEmbedder
from .components.egnn_backbone import EGNNBackbone


class GeoEncoder(nn.Module):
    """
    Geometric Encoder that combines feature embedding and EGNN processing.
    
    This encoder is modality-agnostic. It processes embedded graph data through EGNN layers.
    All modality-specific logic (protein vs molecule) is handled by the FeatureEmbedder.
    
    Processing pipeline:
    1. FeatureEmbedder: Handles modality-specific features and edge concatenation
    2. EGNN Backbone: Modality-agnostic graph neural network with coordinate updates
    
    Args:
        hidden_dim: Hidden dimension for all embeddings and EGNN layers
        num_layers: Number of EGNN layers
        dropout: Dropout probability for EGNN layers
        update_coords: Whether to update coordinates in EGNN layers
        use_layernorm: Whether to use LayerNorm in EGNN layers
        num_rbf: Number of radial basis functions for distance encoding
        rbf_max: Maximum distance for RBF encoding
        protein_residue_id_scale: Scale factor for protein residue ID normalization
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        update_coords: bool = True,
        use_layernorm: bool = True,
        num_rbf: int = 32,
        rbf_max: float = 10.0,
        protein_residue_id_scale: float = 1000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.update_coords = update_coords
        
        # Feature embedder
        self.embedder = FeatureEmbedder(
            hidden_dim=hidden_dim,
            num_rbf=num_rbf,
            rbf_max=rbf_max,
            protein_residue_id_scale=protein_residue_id_scale,
        )
        
        # EGNN backbone
        self.egnn = EGNNBackbone(
            dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            update_coords=update_coords,
            use_layernorm=use_layernorm,
        )
    
    def forward(
        self,
        data: Dict[str, Any],
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through feature embedding and EGNN backbone.
        
        Args:
            data: Dictionary with keys:
                - 'modality': str, either 'protein' or 'molecule'
                - 'value': Dict[str, torch.Tensor], the actual graph data
                  
                  For protein:
                    - node_feat: [N, 7] protein features
                    - edge_attr: [E, 1] edge distances
                    - edge_index: [2, E] edge indices
                    - pos: [N, 3] coordinates
                  
                  For molecule:
                    - node_feat: [N, 9] molecule features
                    - pos: [N, 3] coordinates
                    - edge_index: [2, E_spatial] spatial edges (optional)
                    - edge_feat_dist: [E_spatial, 1] spatial distances (optional)
                    - chem_edge_index: [2, E_chem] chemical edges (optional)
                    - chem_edge_feat_cat: [E_chem, 3] chemical edge features (optional)
            
            batch: Optional [N] batch assignment for each node
        
        Returns:
            Dictionary with keys:
                - node_emb: [N, hidden_dim] final node embeddings
                - pos: [N, 3] updated coordinates (if update_coords=True)
        """
        # Embed graph (modality-specific)
        embedded = self.embedder(data)
        
        # Encode with EGNN (modality-agnostic)
        # EGNN expects: h (node features), pos, edge_index, e (edge features), batch
        node_emb, pos_out = self.egnn(
            h=embedded['node_emb'],
            pos=embedded['pos'],
            edge_index=embedded['edge_index'],
            e=embedded['edge_emb'],
            batch=batch)
        
        return {
            'node_emb': node_emb,
            'pos': pos_out,
        }