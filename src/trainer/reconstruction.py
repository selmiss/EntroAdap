"""
Masked Reconstruction Trainer for Graph Encoder

Wraps AAEncoder with prediction heads for:
1. Node element ID reconstruction
2. Edge distance bin classification (soft targets)
3. Coordinate noise regression (MSE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

from ..models.aa_encoder import AAEncoder


class ReconstructionTrainer(nn.Module):
    """
    Trainer module for masked graph reconstruction.
    
    Assumes data loader provides masks and labels. This module only handles:
    - Forward pass through encoder with masked inputs
    - Prediction heads for reconstruction targets
    - Loss computation with proper normalization
    
    Args:
        encoder: AAEncoder instance
        num_elements: Vocabulary size for element IDs (default: 119, 0-118)
        num_dist_bins: Number of bins for distance discretization
        dist_min: Minimum distance for binning
        dist_max: Maximum distance for binning
        element_weight: Loss weight for element reconstruction
        dist_weight: Loss weight for distance reconstruction
        noise_weight: Loss weight for noise regression
    """
    
    def __init__(
        self,
        encoder: AAEncoder,
        num_elements: int = 119,
        num_dist_bins: int = 64,
        dist_min: float = 0.0,
        dist_max: float = 20.0,
        element_weight: float = 1.0,
        dist_weight: float = 1.0,
        noise_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_elements = num_elements
        self.num_dist_bins = num_dist_bins
        self.element_weight = element_weight
        self.dist_weight = dist_weight
        self.noise_weight = noise_weight
        
        hidden_dim = encoder.hidden_dim
        
        # Prediction heads
        self.element_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_elements)
        )
        
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_dist_bins)
        )
        
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)  # Direct 3D regression
        )
        
        # Bin edges for distance (used for soft targets if needed)
        self.register_buffer('dist_bins', torch.linspace(dist_min, dist_max, num_dist_bins + 1))
    
    def forward(
        self,
        data: Dict[str, Any],
        batch: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        element_labels: Optional[torch.Tensor] = None,
        dist_labels: Optional[torch.Tensor] = None,
        noise_labels: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.
        
        Args:
            data: Graph data dict with 'modality' and 'value' keys
            batch: Optional batch assignment [N]
            node_mask: Boolean mask for masked nodes [N]
            edge_mask: Boolean mask for masked edges [E]
            element_labels: Ground truth element IDs [N] (0-118)
            dist_labels: Ground truth distances [E] or soft targets [E, num_bins]
            noise_labels: Ground truth coordinate noise [N, 3]
            compute_loss: Whether to compute and return losses
        
        Returns:
            Dictionary with:
                - node_emb: [N, hidden_dim]
                - pos: [N, 3]
                - element_logits_masked: [M_nodes, num_elements] (if node_mask provided)
                - dist_logits: [M_edges, num_dist_bins] (if edge_mask provided)
                - noise_pred: [N, 3] (if node_mask provided)
                - num_masked_nodes: int (if node_mask provided)
                - num_masked_edges: int (if edge_mask provided)
                - loss: scalar (if compute_loss=True)
                - element_loss: scalar (if compute_loss=True and node_mask provided)
                - dist_loss: scalar (if compute_loss=True and edge_mask provided)
                - noise_loss: scalar (if compute_loss=True and node_mask provided)
        """
        # Encode
        encoded = self.encoder(data, batch=batch)
        node_emb = encoded['node_emb']  # [N, hidden_dim]
        pos = encoded['pos']  # [N, 3]
        
        result = {
            'node_emb': node_emb,
            'pos': pos,
        }
        
        # Initialize as tensor to ensure backward() works even if no losses are added
        total_loss = torch.tensor(0.0, device=node_emb.device, dtype=node_emb.dtype)
        
        # Element prediction for masked nodes only
        if node_mask is not None:
            num_masked_nodes = node_mask.sum().item()
            result['num_masked_nodes'] = num_masked_nodes
            
            if num_masked_nodes > 0:
                masked_node_emb = node_emb[node_mask]  # [M, hidden_dim]
                element_logits_masked = self.element_head(masked_node_emb)  # [M, num_elements]
                result['element_logits_masked'] = element_logits_masked
                
                if compute_loss and element_labels is not None:
                    masked_labels = element_labels[node_mask]
                    element_loss = F.cross_entropy(element_logits_masked, masked_labels, reduction='mean')
                    result['element_loss'] = element_loss
                    total_loss += self.element_weight * element_loss
        
        # Distance prediction for masked edges only
        if edge_mask is not None:
            num_masked_edges = edge_mask.sum().item()
            result['num_masked_edges'] = num_masked_edges
            
            if num_masked_edges > 0:
                edge_index = data['value']['edge_index']  # [2, E]
                masked_edge_index = edge_index[:, edge_mask]  # [2, M]
                
                src_emb = node_emb[masked_edge_index[0]]  # [M, hidden_dim]
                dst_emb = node_emb[masked_edge_index[1]]  # [M, hidden_dim]
                edge_emb = torch.cat([src_emb, dst_emb], dim=-1)  # [M, hidden_dim * 2]
                
                dist_logits = self.dist_head(edge_emb)  # [M, num_dist_bins]
                result['dist_logits'] = dist_logits
                
                if compute_loss and dist_labels is not None:
                    masked_dist_labels = dist_labels[edge_mask] if dist_labels.dim() == 1 else dist_labels[edge_mask]
                    
                    # Support both hard labels (distances) and soft targets (distributions)
                    if masked_dist_labels.dim() == 1:
                        # Hard labels: convert distances to bin indices
                        dist_bins = self.digitize(masked_dist_labels, self.dist_bins)
                        dist_loss = F.cross_entropy(dist_logits, dist_bins, reduction='mean')
                    else:
                        # Soft targets: use KL divergence with proper normalization
                        t = masked_dist_labels
                        t = t / (t.sum(dim=-1, keepdim=True) + 1e-12)
                        t = t.clamp_min(1e-12)
                        log_probs = F.log_softmax(dist_logits, dim=-1)
                        dist_loss = F.kl_div(log_probs, t, reduction='batchmean')
                    
                    result['dist_loss'] = dist_loss
                    total_loss += self.dist_weight * dist_loss
        
        # Noise prediction for masked nodes (direct 3D regression)
        if node_mask is not None and noise_labels is not None:
            noise_pred = self.noise_head(node_emb)  # [N, 3]
            result['noise_pred'] = noise_pred
            
            if compute_loss:
                masked_pred = noise_pred[node_mask]  # [M, 3]
                masked_labels = noise_labels[node_mask]  # [M, 3]
                num_masked = masked_pred.size(0)
                
                if num_masked > 0:
                    # Per-coordinate mean
                    noise_loss = F.mse_loss(masked_pred, masked_labels, reduction='mean')
                    result['noise_loss'] = noise_loss
                    total_loss += self.noise_weight * noise_loss
        
        if compute_loss:
            result['loss'] = total_loss
        
        return result
    
    def digitize(self, values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous values to bin indices.
        
        Args:
            values: Tensor of any shape
            bins: Bin edges [num_bins + 1]
        
        Returns:
            Bin indices clamped to [0, num_bins-1]
        """
        indices = torch.searchsorted(bins, values.contiguous(), right=False)
        return torch.clamp(indices - 1, 0, len(bins) - 2)
    
    def distances_to_soft_targets(self, distances: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
        """
        Convert distances to soft probability distributions over bins.
        
        Args:
            distances: Ground truth distances [E]
            sigma: Gaussian kernel width for smoothing
        
        Returns:
            Soft targets [E, num_dist_bins]
        """
        # Clamp distances to valid range
        dist_min = self.dist_bins[0]
        dist_max = self.dist_bins[-1]
        distances = torch.clamp(distances, dist_min, dist_max)
        
        bin_centers = (self.dist_bins[:-1] + self.dist_bins[1:]) / 2
        diff = distances.unsqueeze(-1) - bin_centers.unsqueeze(0)  # [E, num_bins]
        soft_targets = torch.exp(-diff ** 2 / (2 * sigma ** 2))
        soft_targets = soft_targets / (soft_targets.sum(dim=-1, keepdim=True) + 1e-12)
        return soft_targets
