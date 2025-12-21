from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

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


@dataclass
class PatchOutput:
    patch_emb: torch.Tensor              # [G, Kmax, D]
    patch_mask: torch.Tensor             # [G, Kmax] True where a patch exists
    anchor_index: torch.Tensor           # [G, Kmax] global node indices (or -1)
    membership: Optional[torch.Tensor]   # [G, Kmax, N] (optional, can be huge)


def soft_patch_grow(
    instr: torch.Tensor,                 # [G, Di]
    x: torch.Tensor,                     # [N, D]
    edge_index: torch.Tensor,            # [2, E]
    batch: torch.Tensor,                 # [N]
    anchor_gate: AnchorGate,
    edge_gate: EdgeGate,
    edge_attr: Optional[torch.Tensor] = None,
    k_max: int = 32,                     # max number of anchors/patches per graph
    r_max: int = 64,                     # max atoms kept per patch (top-r truncation)
    steps: int = 3,                      # growth turns (not hops count in BFS sense, but similar)
    keep_ratio: float = 0.5,             # how much membership stays each turn
    dynamic_k_mass: Optional[float] = None,  # e.g., 0.8 to select anchors until cumulative prob mass
    return_membership: bool = False,
) -> PatchOutput:
    """
    End-to-end differentiable-ish patching:
      1) AnchorGate scores nodes, choose anchors per graph (top-k or mass threshold with cap).
      2) EdgeGate scores edges, convert to expansion weights.
      3) For each anchor, grow soft membership for `steps` turns by diffusing membership along edges.
      4) After each turn, optionally enforce a hard per-patch budget via top-r truncation.
      5) Pool patch embedding as membership-weighted sum of node embeddings.

    Notes:
      - Hard top-k and top-r are non-differentiable. This is still trainable in practice.
        If you want fully soft training, set r_max=None behavior by skipping truncation.
      - This implementation loops over graphs to keep logic simple and correct.

    Returns:
      patch_emb: [G, k_max, D], padded with zeros
      patch_mask: [G, k_max] indicates valid patches
      anchor_index: [G, k_max] global node indices or -1
      membership: optional [G, k_max, N] (can be very memory heavy)
    """
    device = x.device
    G = int(instr.size(0))
    N, D = x.size(0), x.size(1)

    # 1) anchor logits per node
    anchor_logits = anchor_gate(instr, x, batch)  # [N]

    # 2) edge expansion logits per edge
    edge_logits = edge_gate(instr, x, edge_index, batch, edge_attr=edge_attr)  # [E]

    # Convert edge logits to positive weights. Sigmoid is simplest and stable.
    edge_w = torch.sigmoid(edge_logits)  # [E] in (0,1)

    # Prepare outputs
    patch_emb = torch.zeros((G, k_max, D), device=device, dtype=x.dtype)
    patch_mask = torch.zeros((G, k_max), device=device, dtype=torch.bool)
    anchor_index_out = torch.full((G, k_max), -1, device=device, dtype=torch.long)

    membership_out = None
    if return_membership:
        membership_out = torch.zeros((G, k_max, N), device=device, dtype=x.dtype)

    # Helper: pick anchors for one graph
    def _select_anchors(node_ids: torch.Tensor) -> torch.Tensor:
        # node_ids: global node ids for this graph
        logits = anchor_logits[node_ids]
        if logits.numel() == 0:
            return node_ids.new_empty((0,), dtype=torch.long)

        if dynamic_k_mass is None:
            k = min(k_max, logits.numel())
            top = torch.topk(logits, k=k, largest=True).indices
            return node_ids[top]

        # Mass-based: select until cumulative softmax mass reaches threshold, cap at k_max.
        probs = torch.softmax(logits, dim=0)
        vals, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(vals, dim=0)
        k = int((cum < float(dynamic_k_mass)).sum().item()) + 1
        k = max(1, min(k, k_max, probs.numel()))
        chosen_local = idx[:k]
        return node_ids[chosen_local]

    # Main per-graph loop
    src_all, dst_all = edge_index[0], edge_index[1]

    for g in range(G):
        node_ids = torch.nonzero(batch == g, as_tuple=False).view(-1)  # global node ids
        n_g = int(node_ids.numel())
        if n_g == 0:
            continue

        # Select anchors (global ids)
        anchors_global = _select_anchors(node_ids)
        k_g = int(anchors_global.numel())
        if k_g == 0:
            continue

        # Map global node ids to local [0..n_g-1]
        global_to_local = torch.full((N,), -1, device=device, dtype=torch.long)
        global_to_local[node_ids] = torch.arange(n_g, device=device, dtype=torch.long)

        anchors_local = global_to_local[anchors_global]  # [k_g]

        # Extract edges belonging to this graph (both ends in this graph)
        # Assumes batch is per node; keeps intra-graph edges only.
        in_g_src = (batch[src_all] == g)
        in_g_dst = (batch[dst_all] == g)
        e_mask = in_g_src & in_g_dst
        e_ids = torch.nonzero(e_mask, as_tuple=False).view(-1)

        # If no edges, patches are just anchors
        x_g = x[node_ids]  # [n_g, D]

        if e_ids.numel() == 0:
            # Each patch pools just the anchor atom
            for j in range(min(k_g, k_max)):
                a_loc = int(anchors_local[j].item())
                patch_emb[g, j] = x_g[a_loc]
                patch_mask[g, j] = True
                anchor_index_out[g, j] = int(anchors_global[j].item())
                if return_membership:
                    membership_out[g, j, anchors_global[j]] = 1.0
            continue

        # Remap edges to local indices
        src = global_to_local[src_all[e_ids]]  # [E_g]
        dst = global_to_local[dst_all[e_ids]]  # [E_g]
        w = edge_w[e_ids]                      # [E_g]

        # Normalize expansion weights per-source so each node distributes to neighbors stably.
        # If a node has no outgoing edges, it simply does not send.
        denom = torch.zeros((n_g,), device=device, dtype=x.dtype)
        denom.index_add_(0, src, w)
        w_norm = w / (denom[src] + 1e-12)      # [E_g]

        # Initialize membership matrix for all anchors: [k_g, n_g]
        m = torch.zeros((k_g, n_g), device=device, dtype=x.dtype)
        m[torch.arange(k_g, device=device), anchors_local] = 1.0  # one-hot anchors

        # Growth turns
        for _ in range(int(steps)):
            # Send: each anchor row diffuses along edges
            # msgs[:, dst] += m[:, src] * w_norm
            msgs = torch.zeros_like(m)
            msgs.index_add_(1, dst, m[:, src] * w_norm.unsqueeze(0))

            # Keep + send
            m = (float(keep_ratio) * m) + ((1.0 - float(keep_ratio)) * msgs)

            # Enforce per-patch atom budget
            if r_max is not None and r_max < n_g:
                # Keep top-r per patch row
                topv, topi = torch.topk(m, k=int(r_max), dim=1, largest=True)
                m2 = torch.zeros_like(m)
                m2.scatter_(1, topi, topv)
                m = m2

            # Renormalize per patch (mass conserving)
            m = m / (m.sum(dim=1, keepdim=True) + 1e-12)

        # Pool patch embeddings: [k_g, D] = [k_g, n_g] @ [n_g, D]
        p = m @ x_g

        # Write padded outputs
        write_k = min(k_g, k_max)
        patch_emb[g, :write_k] = p[:write_k]
        patch_mask[g, :write_k] = True
        anchor_index_out[g, :write_k] = anchors_global[:write_k]
        if return_membership:
            # store to global-N axis (sparse is better, but this is explicit)
            for j in range(write_k):
                membership_out[g, j, node_ids] = m[j]

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
    E = 400         # edges
    De = 16         # edge attr dim

    instr = torch.randn(G, Di)
    x = torch.randn(N, D)
    batch = torch.randint(0, G, (N,))
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, De)

    anchor_gate = AnchorGate(node_dim=D, instr_dim=Di, hidden_dim=256)
    edge_gate = EdgeGate(node_dim=D, instr_dim=Di, edge_attr_dim=De, hidden_dim=256)

    out = soft_patch_grow(
        instr=instr,
        x=x,
        edge_index=edge_index,
        batch=batch,
        anchor_gate=anchor_gate,
        edge_gate=edge_gate,
        edge_attr=edge_attr,
        k_max=16,
        r_max=64,
        steps=3,
        keep_ratio=0.6,
        dynamic_k_mass=0.8,     # set None for fixed top-k
        return_membership=False
    )

    print(out.patch_emb.shape, out.patch_mask.shape, out.anchor_index.shape)
