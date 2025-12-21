import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .fusion_blocks import FusionBlock
from .geo_encoder import GeoEncoder
from .con_gates import AnchorGate, EdgeGate, soft_patch_grow
from .multimodal_llm_config import MultiModalLLMConfig, BaseConfig


class MultiModalLLM(PreTrainedModel):
    """Multi-modal LLM integrating graph encoder, instruction-conditioned patching, and cross-attention fusion."""
    
    def __init__(
        self,
        llm_model: AutoModelForCausalLM,
        config: Optional[MultiModalLLMConfig] = None,
        **kwargs,
    ):
        """
        Initialize integrated multimodal LLM.
        
        Args:
            llm_model: Pre-loaded language model
            config: MultiModalLLMConfig object (default: BaseConfig())
            **kwargs: Legacy individual params (override config if provided)
        
        Examples:
            # Recommended: Use config
            model = MultiModalLLM(llm_model=llm, config=BaseConfig())
            
            # Custom config
            config = MultiModalLLMConfig(
                encoder=EncoderConfig(hidden_dim=256),
                patching=PatchingConfig(k_max=32),
                fusion=FusionConfig(num_blocks=4),
            )
            model = MultiModalLLM(llm_model=llm, config=config)
            
            # Legacy: Override specific params
            model = MultiModalLLM(llm_model=llm, config=BaseConfig(), k_max=48)
        """
        super().__init__(llm_model.config)
        
        self.llm_model = llm_model
        self.llm_hidden_dim = llm_model.config.hidden_size
        
        # Use provided config or create default
        self.config_mm = config if config is not None else BaseConfig()
        
        # Extract config values
        enc_cfg = self.config_mm.encoder
        patch_cfg = self.config_mm.patching
        fusion_cfg = self.config_mm.fusion
        
        # 1. Graph encoder
        self.encoder = GeoEncoder(
            hidden_dim=enc_cfg.hidden_dim,
            num_layers=enc_cfg.num_layers,
            dropout=enc_cfg.dropout,
            update_coords=enc_cfg.update_coords,
        )
        
        # 2. Instruction projection: LLM hidden -> encoder dim
        self.instr_proj = nn.Linear(self.llm_hidden_dim, enc_cfg.hidden_dim)
        
        # 3. Anchor & Edge gates (accept projected instruction embeddings in encoder space)
        self.anchor_gate = AnchorGate(
            node_dim=enc_cfg.hidden_dim,
            instr_dim=enc_cfg.hidden_dim,
            hidden_dim=patch_cfg.gate_hidden_dim,
            dropout=patch_cfg.gate_dropout,
        )
        self.edge_gate = EdgeGate(
            node_dim=enc_cfg.hidden_dim,
            instr_dim=enc_cfg.hidden_dim,
            edge_attr_dim=enc_cfg.hidden_dim,
            hidden_dim=patch_cfg.gate_hidden_dim,
            dropout=patch_cfg.gate_dropout,
        )
        
        # 4. Fusion projection & blocks
        fusion_hidden = fusion_cfg.hidden_dim if fusion_cfg.hidden_dim is not None else enc_cfg.hidden_dim
        
        # Patch & node projection to fusion space
        self.patch_proj = nn.Linear(enc_cfg.hidden_dim, fusion_hidden) if enc_cfg.hidden_dim != fusion_hidden else nn.Identity()
        self.node_proj = nn.Linear(enc_cfg.hidden_dim, fusion_hidden) if enc_cfg.hidden_dim != fusion_hidden else nn.Identity()
        
        # 5. Fusion blocks (query=patches, KV=nodes)
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(
                hidden_dim=fusion_hidden,
                num_heads=fusion_cfg.num_heads,
                intermediate_dim=fusion_cfg.intermediate_dim,
                dropout=fusion_cfg.dropout,
            )
            for _ in range(fusion_cfg.num_blocks)
        ])
        
        # 6. Output projection to LLM space
        self.output_proj = nn.Linear(fusion_hidden, self.llm_hidden_dim)
        self.output_norm = nn.LayerNorm(self.llm_hidden_dim)
        
    def get_llm_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings from LLM embedding layer."""
        embeds = self.llm_model.get_input_embeddings()(input_ids)
        if not embeds.requires_grad and self.training:
            embeds.requires_grad_(True)
        return embeds
    
    def encode_and_patch(
        self,
        graph_data: Dict[str, Any],
        instr_emb: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode graph and extract instruction-conditioned patches.
        
        Args:
            graph_data: {'modality': str, 'value': dict} with node/edge/pos data
            instr_emb: [G, llm_hidden_dim] instruction embeddings from LLM
            batch: [N] node-to-graph assignment
        
        Returns:
            patch_emb: [G, k_max, enc_dim] patch embeddings
            patch_mask: [G, k_max] valid patch indicator
            node_emb: [N, enc_dim] node embeddings
        """
        # Encode graph
        enc_out = self.encoder(graph_data, batch=batch)
        node_emb = enc_out['node_emb']  # [N, enc_dim]
        edge_emb = enc_out['edge_emb']  # [E, enc_dim]
        
        # Extract edge_index
        edge_index = enc_out['edge_index']
        
        # Project instruction embeddings from LLM space to encoder space
        instr_emb_proj = self.instr_proj(instr_emb)  # [G, enc_dim]
        
        # Run patching with config
        cfg = self.config_mm.patching
        patch_out = soft_patch_grow(
            instr=instr_emb_proj,
            x=node_emb,
            edge_index=edge_index,
            batch=batch,
            anchor_gate=self.anchor_gate,
            edge_gate=self.edge_gate,
            edge_attr=edge_emb,
            k_max=cfg.k_max,
            r_max=cfg.r_max,
            steps=cfg.steps,
            keep_ratio=cfg.keep_ratio,
            dynamic_k_mass=cfg.dynamic_k_mass,
            return_membership=False,
        )
        
        return patch_out.patch_emb, patch_out.patch_mask, node_emb
    
    def fuse_patches_with_nodes(
        self,
        patch_emb: torch.Tensor,
        node_emb: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse patch embeddings (Q) with node embeddings (KV) via cross-attention.
        
        Args:
            patch_emb: [G, k_max, enc_dim] patch embeddings
            node_emb: [N, enc_dim] node embeddings
            patch_mask: [G, k_max] 1=valid patch, 0=padding
            node_mask: [N] 1=valid node, 0=padding (optional)
            batch: [N] node-to-graph assignment (optional, for per-graph KV)
        
        Returns:
            fused_emb: [G, k_max, llm_hidden_dim] fused patch embeddings
        """
        G, k_max, _ = patch_emb.shape
        N = node_emb.size(0)
        
        # Project to fusion space
        patch_h = self.patch_proj(patch_emb)  # [G, k_max, fusion_dim]
        
        # For cross-attention KV: replicate nodes per graph or use all nodes
        # Simple approach: each graph attends to ALL nodes (batch-agnostic)
        # Better: per-graph KV using batch index
        if batch is not None:
            # Build per-graph node KV: [G, N_g, fusion_dim]
            # For simplicity here, we'll stack all nodes and let mask handle it
            # (more efficient impl would gather per-graph nodes)
            node_h = self.node_proj(node_emb).unsqueeze(0).expand(G, -1, -1)  # [G, N, fusion_dim]
            
            # Build KV mask: [G, N] where 1=this node belongs to graph g
            kv_mask = torch.zeros(G, N, dtype=torch.bool, device=node_emb.device)
            for g in range(G):
                kv_mask[g, batch == g] = True
            if node_mask is not None:
                kv_mask = kv_mask & node_mask.bool().unsqueeze(0)
        else:
            # No batch info: all nodes attend to all nodes
            node_h = self.node_proj(node_emb).unsqueeze(0).expand(G, -1, -1)
            kv_mask = None
        
        # Prepare masks for MultiheadAttention (True=padding, False=valid)
        q_mask = None
        if patch_mask is not None:
            q_mask = ~patch_mask.bool()  # invert
        
        kv_padding_mask = None
        if kv_mask is not None:
            kv_padding_mask = ~kv_mask  # invert
        
        # Apply fusion blocks
        hidden = patch_h
        for fusion_block in self.fusion_blocks:
            hidden = fusion_block(
                hidden_states=hidden,
                key_value_states=node_h,
                key_padding_mask=q_mask,
                cross_key_padding_mask=kv_padding_mask,
            )
        
        # Project to LLM space
        fused = self.output_proj(hidden)  # [G, k_max, llm_hidden_dim]
        fused = self.output_norm(fused)
        
        return fused
    
    def forward(
        self,
        input_ids: torch.Tensor,
        graph_data: Optional[Dict[str, Any]] = None,
        batch: Optional[torch.Tensor] = None,
        instr_positions: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass integrating graph encoder, patching, and LLM.
        
        Args:
            input_ids: [B, seq_len] token IDs
            graph_data: {'modality': str, 'value': {...}} graph input
            batch: [N] node-to-graph assignment
            instr_positions: [B, n_instr] token positions that contain instructions
            patch_positions: [B, k_max] token positions to inject patches (optional)
            attention_mask: [B, seq_len] attention mask
            patch_mask: [B, k_max] valid patch mask (optional)
            node_mask: [N] valid node mask (optional)
            labels: [B, seq_len] training labels
            return_dict: whether to return dict
        
        Returns:
            CausalLMOutputWithPast
        """
        # Get text embeddings
        inputs_embeds = self.get_llm_embeddings(input_ids)  # [B, seq_len, llm_hidden_dim]
        B, seq_len, _ = inputs_embeds.shape
        
        # If graph data provided, encode and fuse
        if graph_data is not None and batch is not None and instr_positions is not None:
            # Extract instruction embeddings from specific token positions
            # instr_positions: [B, n_instr], we pool or take first/last
            # For simplicity: take mean of instruction tokens per sample
            G = batch.max().item() + 1
            instr_emb_list = []
            for g in range(G):
                if g < B:
                    instr_pos = instr_positions[g]
                    valid_pos = instr_pos[instr_pos >= 0]
                    valid_pos = valid_pos[valid_pos < seq_len]
                    if valid_pos.numel() > 0:
                        instr_tokens = inputs_embeds[g, valid_pos]  # [n_instr, llm_hidden_dim]
                        instr_emb_list.append(instr_tokens.mean(dim=0))
                    else:
                        instr_emb_list.append(torch.zeros(self.llm_hidden_dim, device=inputs_embeds.device))
                else:
                    instr_emb_list.append(torch.zeros(self.llm_hidden_dim, device=inputs_embeds.device))
            instr_emb = torch.stack(instr_emb_list, dim=0)  # [G, llm_hidden_dim]
            
            # Encode and patch
            patch_emb, out_patch_mask, node_emb = self.encode_and_patch(
                graph_data, instr_emb, batch
            )  # [G, k_max, enc_dim], [G, k_max], [N, enc_dim]

            # Optional external patch mask (e.g., user-provided valid patch slots)
            effective_patch_mask = out_patch_mask
            if patch_mask is not None:
                effective_patch_mask = effective_patch_mask & patch_mask[: effective_patch_mask.size(0)].bool()
            
            # Fuse patches with nodes
            fused_patches = self.fuse_patches_with_nodes(
                patch_emb, node_emb, effective_patch_mask, node_mask, batch
            )  # [G, k_max, llm_hidden_dim]
            
            # Inject patches into text sequence
            if patch_positions is not None:
                # Replace embeddings at specified positions
                k_max = self.config_mm.patching.k_max
                for g in range(min(G, B)):
                    for k in range(k_max):
                        pos = patch_positions[g, k]
                        if pos >= 0 and pos < seq_len and (effective_patch_mask is None or effective_patch_mask[g, k]):
                            inputs_embeds[g, pos] = fused_patches[g, k]
            else:
                # Concatenate at beginning
                inputs_embeds = torch.cat([fused_patches[:B], inputs_embeds], dim=1)
                
                # Adjust masks and labels
                k_max = self.config_mm.patching.k_max
                if attention_mask is not None:
                    patch_attn_mask = torch.ones(
                        (B, k_max), dtype=attention_mask.dtype, device=attention_mask.device
                    )
                    if effective_patch_mask is not None:
                        patch_attn_mask = effective_patch_mask[:B].to(attention_mask.dtype)
                    attention_mask = torch.cat([patch_attn_mask, attention_mask], dim=1)
                
                if labels is not None:
                    ignore_labels = torch.full(
                        (B, k_max), -100, dtype=labels.dtype, device=labels.device
                    )
                    labels = torch.cat([ignore_labels, labels], dim=1)
        
        # Forward through LLM
        return self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        graph_data: Optional[Dict[str, Any]] = None,
        batch: Optional[torch.Tensor] = None,
        instr_positions: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate with graph context."""
        self.eval()
        
        inputs_embeds = self.get_llm_embeddings(input_ids)
        B, seq_len, _ = inputs_embeds.shape
        
        if graph_data is not None and batch is not None and instr_positions is not None:
            G = batch.max().item() + 1
            instr_emb_list = []
            for g in range(G):
                if g < B:
                    instr_pos = instr_positions[g]
                    valid_pos = instr_pos[instr_pos >= 0]
                    valid_pos = valid_pos[valid_pos < seq_len]
                    if valid_pos.numel() > 0:
                        instr_tokens = inputs_embeds[g, valid_pos]
                        instr_emb_list.append(instr_tokens.mean(dim=0))
                    else:
                        instr_emb_list.append(torch.zeros(self.llm_hidden_dim, device=inputs_embeds.device))
                else:
                    instr_emb_list.append(torch.zeros(self.llm_hidden_dim, device=inputs_embeds.device))
            instr_emb = torch.stack(instr_emb_list, dim=0)
            
            patch_emb, out_patch_mask, node_emb = self.encode_and_patch(graph_data, instr_emb, batch)

            effective_patch_mask = out_patch_mask
            if patch_mask is not None:
                effective_patch_mask = effective_patch_mask & patch_mask[: effective_patch_mask.size(0)].bool()

            fused_patches = self.fuse_patches_with_nodes(patch_emb, node_emb, effective_patch_mask, node_mask, batch)
            
            k_max = self.config_mm.patching.k_max
            if patch_positions is not None:
                for g in range(min(G, B)):
                    for k in range(k_max):
                        pos = patch_positions[g, k]
                        if pos >= 0 and pos < seq_len and (effective_patch_mask is None or effective_patch_mask[g, k]):
                            inputs_embeds[g, pos] = fused_patches[g, k]
            else:
                inputs_embeds = torch.cat([fused_patches[:B], inputs_embeds], dim=1)
                if attention_mask is not None:
                    patch_attn_mask = torch.ones((B, k_max), dtype=attention_mask.dtype, device=attention_mask.device)
                    if effective_patch_mask is not None:
                        patch_attn_mask = effective_patch_mask[:B].to(attention_mask.dtype)
                    attention_mask = torch.cat([patch_attn_mask, attention_mask], dim=1)
        
        return self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation - delegates to underlying LLM."""
        return self.llm_model.prepare_inputs_for_generation(input_ids, **kwargs)
    
    def can_generate(self):
        """Check if model can generate - delegates to underlying LLM."""
        if hasattr(self, 'llm_model'):
            return self.llm_model.can_generate()
        return True
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search - delegates to underlying LLM."""
        return self.llm_model._reorder_cache(past_key_values, beam_idx)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing - delegates to underlying LLM."""
        if hasattr(self.llm_model, 'gradient_checkpointing_enable'):
            self.llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing - delegates to underlying LLM."""
        if hasattr(self.llm_model, 'gradient_checkpointing_disable'):
            self.llm_model.gradient_checkpointing_disable()  
        else:
            super().gradient_checkpointing_disable()
    
    def enable_multimodal_training(self):
        """Ensure multimodal components are trainable (call after applying PEFT)."""
        trainable_modules = [
            self.encoder,
            self.instr_proj,
            self.anchor_gate,
            self.edge_gate,
            self.patch_proj,
            self.node_proj,
            self.fusion_blocks,
            self.output_proj,
            self.output_norm,
        ]
        
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
