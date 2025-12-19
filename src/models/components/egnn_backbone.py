"""
EGNN Backbone Implementation

E(n)-Equivariant Graph Neural Network backbone for processing graph-structured data
with both node features and 3D coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int) -> torch.Tensor:
    """
    Scatter add operation using native PyTorch.
    
    Aggregates values from src to positions specified by index along dimension dim.
    
    Args:
        src: Source tensor to scatter [E, ...]
        index: Indices where to scatter [E]
        dim: Dimension along which to scatter (should be 0)
        dim_size: Size of output dimension
    
    Returns:
        Scattered tensor [dim_size, ...]
    """
    assert dim == 0, "Only dim=0 is supported"
    
    # Create output tensor with same dtype and device as src
    shape = list(src.shape)
    shape[0] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    
    # Use index_add_ for efficient aggregation
    out.index_add_(0, index, src)
    
    return out


class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers and dropout."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EGNNLayer(nn.Module):
    """
    Single E(n)-Equivariant Graph Neural Network layer.
    
    Performs message passing with optional coordinate updates while maintaining
    E(n)-equivariance (invariance to rotations and translations).
    """
    
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        update_coords: bool = True,
        use_layernorm: bool = True,
        eps: float = 1e-8
    ):
        super().__init__()
        self.dim = dim
        self.update_coords = update_coords
        self.eps = eps
        
        # Message MLP: takes [h_src, h_dst, e, dij]
        self.mlp_msg = MLP(
            in_dim=dim + dim + dim + 1,  # h_src + h_dst + e + distance
            out_dim=dim,
            hidden_dim=dim * 2,
            num_layers=2,
            dropout=dropout
        )
        
        # Coordinate update MLP (if enabled)
        if self.update_coords:
            self.mlp_coord = MLP(
                in_dim=dim,
                out_dim=1,
                hidden_dim=dim,
                num_layers=2,
                dropout=dropout
            )
        
        # Node update MLP: takes [h, m_i]
        self.mlp_node = MLP(
            in_dim=dim + dim,  # h + aggregated_message
            out_dim=dim,
            hidden_dim=dim * 2,
            num_layers=2,
            dropout=dropout
        )
        
        # Optional LayerNorm
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of EGNN layer.
        
        Args:
            h: Node features [N, dim]
            pos: Node coordinates [N, 3]
            edge_index: Edge indices [2, E].
                **Convention in this implementation**:
                - edge_index[0] = receiver / node being updated (i)
                - edge_index[1] = neighbor / sender (j)
                Messages and coordinate updates are aggregated to edge_index[0].
            e: Edge features [E, dim]
        
        Returns:
            h_out: Updated node features [N, dim]
            pos_out: Updated coordinates [N, 3]
        """
        N = h.size(0)

        # PyG convention: edge_index[0]=sender/source, edge_index[1]=receiver/target
        send, recv = edge_index[0], edge_index[1]
        
        # Compute relative vectors and distances (x_i - x_j for each (i, j) edge)
        rij = pos[recv] - pos[send]  # [E, 3]
        dij_sq = torch.sum(rij ** 2, dim=1, keepdim=True)  # [E, 1]
        dij = torch.sqrt(dij_sq + self.eps)  # [E, 1]
        
        # Compute messages
        msg_input = torch.cat([h[recv], h[send], e, dij], dim=1)  # [E, 3*dim + 1]
        m_ij = self.mlp_msg(msg_input)  # [E, dim]
        
        # Optional coordinate update
        if self.update_coords:
            # Compute coordinate scaling factors
            s_ij = self.mlp_coord(m_ij)  # [E, 1]
            
            # Compute coordinate updates (E(n)-equivariant)
            coord_diff = rij / (dij + self.eps) * s_ij  # [E, 3]
            
            # Aggregate coordinate updates
            delta_pos = scatter_add(coord_diff, recv, dim=0, dim_size=N)
            
            pos = pos + delta_pos  # [N, 3]
        
        # Aggregate messages
        m_i = scatter_add(m_ij, recv, dim=0, dim_size=N)  # [N, dim]
        
        # Update node features (residual connection)
        node_input = torch.cat([h, m_i], dim=1)  # [N, 2*dim]
        h = h + self.mlp_node(node_input)  # [N, dim]
        
        # Optional LayerNorm
        if self.use_layernorm:
            h = self.layer_norm(h)
        
        return h, pos


class EGNNBackbone(nn.Module):
    """
    EGNN Backbone: Stack of E(n)-Equivariant Graph Neural Network layers.
    
    Processes graph-structured data with node features, coordinates, and edge features.
    Maintains E(n)-equivariance throughout the network.
    
    Args:
        dim: Feature dimension for nodes and edges
        num_layers: Number of EGNN layers to stack
        dropout: Dropout probability for MLPs
        update_coords: Whether to update coordinates in each layer
        use_layernorm: Whether to apply LayerNorm after each layer
    """
    
    def __init__(
        self,
        dim: int,
        num_layers: int = 6,
        dropout: float = 0.1,
        update_coords: bool = True,
        use_layernorm: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.update_coords = update_coords
        
        # Stack of EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(
                dim=dim,
                dropout=dropout,
                update_coords=update_coords,
                use_layernorm=use_layernorm
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through EGNN backbone.
        
        Args:
            h: Node features [N, dim]
            pos: Node coordinates [N, 3]
            edge_index: Edge indices [2, E].
                **Convention in this implementation**:
                - edge_index[0] = receiver / node being updated (i)
                - edge_index[1] = neighbor / sender (j)
            e: Edge features [E, dim]
            batch: Optional batch assignment for each node [N] (not used in computation, kept for API compatibility)
        
        Returns:
            h_out: Updated node features [N, dim]
            pos_out: Updated coordinates [N, 3] (or unchanged pos if update_coords=False)
        """
        # Apply each EGNN layer sequentially
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, e)
        
        return h, pos
