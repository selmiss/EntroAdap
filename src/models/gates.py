from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class AnchorGate(nn.Module):
    """
    Scores each node as an anchor candidate, conditioned on an instruction vector.

    Inputs:
      instr: [G, Di] instruction embeddings (one per graph)
      x:     [N, D]  node embeddings (batched graphs)
      batch: [N]     graph id per node in [0, G-1]

    Output:
      anchor_logits: [N] unnormalized scores
    """
    def __init__(self, node_dim: int, instr_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = _mlp(node_dim + instr_dim, hidden_dim, 1, dropout=dropout)

    def forward(self, instr: torch.Tensor, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        u = instr[batch]                              # [N, Di]
        h = torch.cat([x, u], dim=-1)                 # [N, D+Di]
        return self.net(h).squeeze(-1)                # [N]
    
    def freeze(self):
        """Freeze all parameters in the anchor gate."""
        for param in self.parameters():
            param.requires_grad = False


class EdgeGate(nn.Module):
    """
    Scores each edge for expansion, conditioned on instruction.

    Inputs:
      instr:     [G, Di]
      x:         [N, D]
      edge_index:[2, E] (src, dst)
      edge_attr: [E, De] (optional)
      batch:     [N] graph id per node

    Output:
      edge_logits: [E] unnormalized scores (higher = easier to expand through)
    """
    def __init__(
        self,
        node_dim: int,
        instr_dim: int,
        edge_attr_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = (2 * node_dim) + instr_dim + edge_attr_dim
        self.net = _mlp(in_dim, hidden_dim, 1, dropout=dropout)
        self.edge_attr_dim = edge_attr_dim

    def forward(
        self,
        instr: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        u = instr[batch[src]]                         # [E, Di]
        parts = [x[src], x[dst], u]
        if self.edge_attr_dim > 0:
            assert edge_attr is not None, "edge_attr_dim>0 but edge_attr is None"
            parts.append(edge_attr)
        h = torch.cat(parts, dim=-1)                  # [E, ...]
        return self.net(h).squeeze(-1)                # [E]
    
    def freeze(self):
        """Freeze all parameters in the edge gate."""
        for param in self.parameters():
            param.requires_grad = False


@dataclass
class PatchOutput:
    patch_emb: torch.Tensor              # [G, Kmax, D]
    patch_mask: torch.Tensor             # [G, Kmax] True where a patch exists
    anchor_index: torch.Tensor           # [G, Kmax] global node indices (or -1)
    membership: Optional[torch.Tensor]   # [G, Kmax, N] (optional, can be huge)


def soft_patch_grow(
    instr: torch.Tensor,                 # [G, Di]
    x: torch.Tensor,                     # [N, D]
    pos: torch.Tensor,                   # [N, 3]
    batch: torch.Tensor,                 # [N]
    anchor_gate: AnchorGate,
    k_max: int = 32,
    r_max: Optional[int] = 64,           # top-r nodes per patch (optional, nondifferentiable)
    dynamic_k_mass: Optional[float] = 0.8,
    beta: Optional[float] = None,        # distance scale; if None uses 1.0
    tau: float = 0.1,                    # softmax temperature
    use_anchor_bias: bool = True,        # add anchor logit bias to each patch
    return_membership: bool = False,
) -> PatchOutput:
    """
    Differentiable patching without BFS:
      1) AnchorGate scores nodes, select anchors per graph (top-k or mass threshold).
      2) Assign every node to anchors with soft weights based on Euclidean distance in pos.
      3) Pool patch tokens as assignment-weighted averages of node embeddings.

    Notes:
      - Anchor selection is still discrete (top-k or mass). Gradients flow to selected anchors.
      - If r_max is not None, per-patch truncation via top-k is nondifferentiable. Set r_max=None for fully soft pooling.
    """
    device = x.device
    G = int(instr.size(0))
    N, D = x.size(0), x.size(1)

    if beta is None:
        beta = 1.0

    # 1) anchor logits per node
    anchor_logits = anchor_gate(instr, x, batch)  # [N]

    patch_emb = torch.zeros((G, k_max, D), device=device, dtype=x.dtype)
    patch_mask = torch.zeros((G, k_max), device=device, dtype=torch.bool)
    anchor_index_out = torch.full((G, k_max), -1, device=device, dtype=torch.long)

    membership_out = None
    if return_membership:
        membership_out = torch.zeros((G, k_max, N), device=device, dtype=x.dtype)

    actual_max_k = 0

    def _select_anchors(node_ids: torch.Tensor) -> torch.Tensor:
        logits = anchor_logits[node_ids]
        if logits.numel() == 0:
            return node_ids.new_empty((0,), dtype=torch.long)

        if dynamic_k_mass is None:
            k = min(k_max, logits.numel())
            top = torch.topk(logits, k=k, largest=True).indices
            return node_ids[top]

        probs = torch.softmax(logits, dim=0)
        vals, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(vals, dim=0)
        k = int((cum < float(dynamic_k_mass)).sum().item()) + 1
        k = max(1, min(k, k_max, probs.numel()))
        chosen_local = idx[:k]
        return node_ids[chosen_local]

    eps = 1e-12

    for g in range(G):
        node_ids = torch.nonzero(batch == g, as_tuple=False).view(-1)  # global node ids
        n_g = int(node_ids.numel())
        if n_g == 0:
            continue

        anchors_global = _select_anchors(node_ids)
        k_g = int(anchors_global.numel())
        if k_g == 0:
            continue

        x_g = x[node_ids]       # [n_g, D]
        pos_g = pos[node_ids]   # [n_g, 3]

        # Map global anchor ids to local indices in [0, n_g)
        node_ids_sorted, perm = torch.sort(node_ids)
        pos_in_sorted = torch.searchsorted(node_ids_sorted, anchors_global)
        anchors_local = perm[pos_in_sorted]  # [k_g]

        anchor_pos = pos_g[anchors_local]  # [k_g, 3]
        anchor_bias = anchor_logits[anchors_global] if use_anchor_bias else None  # [k_g] or None

        # 2) soft assignment A: [n_g, k_g]
        dist2 = ((pos_g[:, None, :] - anchor_pos[None, :, :]) ** 2).sum(dim=-1)  # [n_g, k_g]
        scores = (-float(beta) * dist2)
        if anchor_bias is not None:
            scores = scores + anchor_bias[None, :]

        A = torch.softmax(scores / max(float(tau), 1e-8), dim=1)  # [n_g, k_g], rows sum to 1

        # Optional per-patch top-r truncation (keeps most contributing nodes per patch)
        if r_max is not None and int(r_max) < n_g:
            AT = A.transpose(0, 1).contiguous()  # [k_g, n_g]
            topv, topi = torch.topk(AT, k=int(r_max), dim=1, largest=True)
            AT2 = torch.zeros_like(AT)
            AT2.scatter_(1, topi, topv)
            # Normalize per patch for pooling stability
            AT2 = AT2 / (AT2.sum(dim=1, keepdim=True) + eps)
            A_for_pool = AT2.transpose(0, 1)  # [n_g, k_g]
        else:
            # Normalize per patch for pooling stability
            mass = A.sum(dim=0, keepdim=True)  # [1, k_g]
            A_for_pool = A / (mass + eps)

        # 3) pool: tokens [k_g, D]
        tokens = A_for_pool.transpose(0, 1) @ x_g  # [k_g, D]

        write_k = min(k_g, k_max)
        patch_emb[g, :write_k] = tokens[:write_k]
        patch_mask[g, :write_k] = True
        anchor_index_out[g, :write_k] = anchors_global[:write_k]

        if return_membership:
            # Store normalized per-patch weights over global N
            # membership_out[g, j, node_ids] = weights over nodes for patch j
            W = A_for_pool.transpose(0, 1)  # [k_g, n_g]
            membership_out[g, :write_k, node_ids] = W[:write_k]

        actual_max_k = max(actual_max_k, write_k)

    actual_max_k = max(1, actual_max_k)
    patch_emb = patch_emb[:, :actual_max_k, :]
    patch_mask = patch_mask[:, :actual_max_k]
    anchor_index_out = anchor_index_out[:, :actual_max_k]
    if return_membership:
        membership_out = membership_out[:, :actual_max_k, :]

    return PatchOutput(
        patch_emb=patch_emb,
        patch_mask=patch_mask,
        anchor_index=anchor_index_out,
        membership=membership_out if return_membership else None,
    )


# -------------------------
# Example usage (PyG-style)
# -------------------------
if __name__ == "__main__":
    # Dummy shapes
    G = 2           # number of graphs in batch
    N = 100         # total nodes
    D = 128         # node embed dim
    Di = 256        # instruction dim
    instr = torch.randn(G, Di)
    x = torch.randn(N, D)
    batch = torch.randint(0, G, (N,))
    pos = torch.randn(N, 3)

    anchor_gate = AnchorGate(node_dim=D, instr_dim=Di, hidden_dim=256)
    out = soft_patch_grow(
        instr=instr,
        x=x,
        pos=pos,
        batch=batch,
        anchor_gate=anchor_gate,
        k_max=16,
        r_max=64,
        dynamic_k_mass=0.8,     # set None for fixed top-k
        beta=1.0,
        tau=0.1,
        return_membership=False
    )

    print(out.patch_emb.shape, out.patch_mask.shape, out.anchor_index.shape)
