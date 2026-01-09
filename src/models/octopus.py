import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .cross_attn import FusionBlock
from .aa_encoder import AAEncoder
from .gates import AnchorGate, EdgeGate, soft_patch_grow
from .octopus_config import OctopusConfig, BaseConfig


class Octopus(PreTrainedModel):
    """Octopus: Multi-modal LLM integrating graph encoder, instruction-conditioned patching, and cross-attention fusion."""

    # This custom wrapper architecture does not implement SDPA/FlashAttention kernels.
    # Tell Transformers to fall back to eager attention to avoid init-time errors.
    _supports_sdpa = False
    _supports_flash_attn_2 = False
    _supports_flex_attn = False
    
    def __init__(
        self,
        llm_model: AutoModelForCausalLM,
        config: Optional[OctopusConfig] = None,
        **kwargs,
    ):
        """
        Initialize Octopus model.
        
        Args:
            llm_model: Pre-loaded language model
            config: OctopusConfig object (default: BaseConfig())
            **kwargs: Legacy individual params (override config if provided)
        
        Examples:
            # Recommended: Use config
            model = Octopus(llm_model=llm, config=BaseConfig())
            
            # Custom config
            config = OctopusConfig(
                encoder=EncoderConfig(hidden_dim=256),
                patching=PatchingConfig(k_max=32),
                fusion=FusionConfig(num_blocks=4),
            )
            model = Octopus(llm_model=llm, config=config)
            
            # Legacy: Override specific params
            model = Octopus(llm_model=llm, config=BaseConfig(), k_max=48)
        """
        # Ensure the wrapper itself doesn't try to opt into SDPA/FlashAttention.
        try:
            llm_model.config._attn_implementation_internal = "eager"
        except Exception:
            pass

        super().__init__(llm_model.config)
        
        self.llm_model = llm_model
        self.llm_hidden_dim = llm_model.config.hidden_size
        
        # Use provided config or create default
        self.config_octopus = config if config is not None else BaseConfig()
        
        # Store padding side (can be overridden via kwargs, defaults to "right")
        # This should match the tokenizer's padding_side for correct alignment
        self.padding_side = kwargs.get('padding_side', 'right')
        
        # Extract config values
        enc_cfg = self.config_octopus.encoder
        patch_cfg = self.config_octopus.patching
        fusion_cfg = self.config_octopus.fusion
        
        # 1. Graph encoder
        self.encoder = AAEncoder(
            hidden_dim=enc_cfg.hidden_dim,
            num_layers=enc_cfg.num_layers,
            dropout=enc_cfg.dropout,
            update_coords=enc_cfg.update_coords,
        )
        
        # 2. Instruction projection: LLM hidden -> encoder dim
        # Add normalization before projection to stabilize gradients
        self.instr_norm = nn.LayerNorm(self.llm_hidden_dim)
        self.instr_proj = nn.Linear(self.llm_hidden_dim, enc_cfg.hidden_dim)
        # Initialize with small weights for large downprojection (4096->256)
        self._init_projection(self.instr_proj, is_downprojection=True)
        
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
        if enc_cfg.hidden_dim != fusion_hidden:
            # Add normalization before upprojection
            self.patch_norm_pre = nn.LayerNorm(enc_cfg.hidden_dim)
            self.node_norm_pre = nn.LayerNorm(enc_cfg.hidden_dim)
            self.patch_proj = nn.Linear(enc_cfg.hidden_dim, fusion_hidden)
            self.node_proj = nn.Linear(enc_cfg.hidden_dim, fusion_hidden)
            # Initialize with scaled weights for large upprojection (256->4096)
            self._init_projection(self.patch_proj, is_downprojection=False)
            self._init_projection(self.node_proj, is_downprojection=False)
        else:
            self.patch_norm_pre = nn.Identity()
            self.node_norm_pre = nn.Identity()
            self.patch_proj = nn.Identity()
            self.node_proj = nn.Identity()
        
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
        # Initialize output projection with small weights
        self._init_projection(self.output_proj, is_downprojection=False)
        self.output_norm = nn.LayerNorm(self.llm_hidden_dim)
    
    def _init_projection(self, linear_layer: nn.Linear, is_downprojection: bool = False):
        """
        Initialize projection layers with scaled weights to prevent gradient explosion.
        
        Uses Xavier/Glorot initialization with additional scaling for large dimension changes.
        For downprojections (e.g., 4096->256): use smaller init for stability.
        For upprojections (e.g., 256->4096): use smaller init and scale output.
        
        Args:
            linear_layer: The linear layer to initialize
            is_downprojection: True for downprojections, False for upprojections
        """
        # Xavier/Glorot uniform initialization
        nn.init.xavier_uniform_(linear_layer.weight)
        
        # Additional scaling based on dimension ratio
        in_dim = linear_layer.weight.size(1)
        out_dim = linear_layer.weight.size(0)
        dim_ratio = max(in_dim, out_dim) / min(in_dim, out_dim)
        
        # Scale down weights for large dimension changes
        if dim_ratio > 4.0:  # Significant dimension change (e.g., 256<->4096 is 16x)
            scale = 0.02  # Conservative scaling
        elif dim_ratio > 2.0:
            scale = 0.1
        else:
            scale = 1.0
        
        with torch.no_grad():
            linear_layer.weight.mul_(scale)
            if linear_layer.bias is not None:
                linear_layer.bias.zero_()

    # ---------------------------------------------------------------------
    # HuggingFace PreTrainedModel embedding / resizing delegation
    #
    # This wrapper stores the real language model in `self.llm_model`. Many
    # library utilities (tokenizer special-token handling, trainer utilities,
    # etc.) call these methods on the *outer* model. We delegate them to the
    # inner LLM so both standard LLM and wrapped mLLM behave consistently.
    # ---------------------------------------------------------------------

    def get_input_embeddings(self):
        return self.llm_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.llm_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        if hasattr(self.llm_model, "get_output_embeddings"):
            return self.llm_model.get_output_embeddings()
        return None

    def set_output_embeddings(self, new_embeddings):
        if hasattr(self.llm_model, "set_output_embeddings"):
            return self.llm_model.set_output_embeddings(new_embeddings)
        return None

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ):
        """
        Resize the inner LLM token embeddings, and keep this wrapper's config in sync.
        """
        out = self.llm_model.resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )
        # Keep outer config aligned with the underlying LLM config
        try:
            self.config.vocab_size = self.llm_model.config.vocab_size
        except Exception:
            pass
        return out
        
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
        # Apply normalization before projection to stabilize gradients
        instr_emb_norm = self.instr_norm(instr_emb)  # [G, llm_hidden_dim]
        instr_emb_proj = self.instr_proj(instr_emb_norm)  # [G, enc_dim]
        
        # Run patching with config
        cfg = self.config_octopus.patching
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
        
        # Project to fusion space with pre-normalization
        patch_h = self.patch_norm_pre(patch_emb)  # [G, k_max, enc_dim]
        patch_h = self.patch_proj(patch_h)  # [G, k_max, fusion_dim]
        
        # For cross-attention KV: replicate nodes per graph or use all nodes
        # Simple approach: each graph attends to ALL nodes (batch-agnostic)
        # Better: per-graph KV using batch index
        if batch is not None:
            # Build per-graph node KV: [G, N_g, fusion_dim]
            # For simplicity here, we'll stack all nodes and let mask handle it
            # (more efficient impl would gather per-graph nodes)
            node_h = self.node_norm_pre(node_emb)  # [N, enc_dim]
            node_h = self.node_proj(node_h).unsqueeze(0).expand(G, -1, -1)  # [G, N, fusion_dim]
            
            # Build KV mask: [G, N] where 1=this node belongs to graph g
            kv_mask = torch.zeros(G, N, dtype=torch.bool, device=node_emb.device)
            for g in range(G):
                kv_mask[g, batch == g] = True
            if node_mask is not None:
                kv_mask = kv_mask & node_mask.bool().unsqueeze(0)
        else:
            # No batch info: all nodes attend to all nodes
            node_h = self.node_norm_pre(node_emb)  # [N, enc_dim]
            node_h = self.node_proj(node_h).unsqueeze(0).expand(G, -1, -1)
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
    
    def _inject_patches_into_sequence(
        self,
        inputs_embeds: torch.Tensor,
        patches: torch.Tensor,
        patch_positions: torch.Tensor,
        patch_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Insert patches into the embedding sequence at specified positions.
        
        This function INSERTS patches (not replaces), shifting subsequent tokens to the right.
        All related tensors (attention_mask, labels) are adjusted accordingly.
        
        Args:
            inputs_embeds: [B, seq_len, hidden_dim] input embeddings
            patches: [B, k_max, hidden_dim] patch embeddings to inject
            patch_positions: [B, 1] position where to insert patches (single position per sample)
            patch_mask: [B, k_max] valid patch mask
            attention_mask: [B, seq_len] attention mask (optional)
            labels: [B, seq_len] labels (optional)
        
        Returns:
            (new_inputs_embeds, new_attention_mask, new_labels)
            - new_inputs_embeds: [B, seq_len + k_max, hidden_dim]
            - new_attention_mask: [B, seq_len + k_max] (if provided)
            - new_labels: [B, seq_len + k_max] (if provided)
        """
        B, seq_len, hidden_dim = inputs_embeds.shape
        k_max = patches.shape[1]
        device = inputs_embeds.device
        
        # Process each sample in the batch
        new_embeds_list = []
        new_attn_list = [] if attention_mask is not None else None
        new_labels_list = [] if labels is not None else None
        
        for b in range(B):
            pos = patch_positions[b, 0].item()
            
            if pos < 0:
                # No injection for this sample, just keep original + pad to match new length
                # Use model's padding_side to maintain consistency with tokenizer
                pad_embeds = torch.zeros(k_max, hidden_dim, device=device)
                
                if self.padding_side == "left":
                    # Left padding: add new padding on the left
                    new_embeds_list.append(torch.cat([pad_embeds, inputs_embeds[b]], dim=0))
                    if attention_mask is not None:
                        new_attn_list.append(torch.cat([
                            torch.zeros(k_max, dtype=attention_mask.dtype, device=device),
                            attention_mask[b]
                        ], dim=0))
                    if labels is not None:
                        new_labels_list.append(torch.cat([
                            torch.full((k_max,), -100, dtype=labels.dtype, device=device),
                            labels[b]
                        ], dim=0))
                else:
                    # Right padding: add new padding on the right (default behavior)
                    new_embeds_list.append(torch.cat([inputs_embeds[b], pad_embeds], dim=0))
                    if attention_mask is not None:
                        new_attn_list.append(torch.cat([
                            attention_mask[b],
                            torch.zeros(k_max, dtype=attention_mask.dtype, device=device)
                        ], dim=0))
                    if labels is not None:
                        new_labels_list.append(torch.cat([
                            labels[b],
                            torch.full((k_max,), -100, dtype=labels.dtype, device=device)
                        ], dim=0))
            else:
                # Insert patches at position pos
                pos = min(pos, seq_len)  # Clamp to valid range
                
                # Split: [0:pos] + patches + [pos:]
                before = inputs_embeds[b, :pos]
                after = inputs_embeds[b, pos:]
                
                # Get valid patches for this sample
                valid_patches = patches[b]  # [k_max, hidden_dim]
                if patch_mask is not None:
                    # Zero out invalid patches
                    mask_expanded = patch_mask[b].unsqueeze(-1).float()  # [k_max, 1]
                    valid_patches = valid_patches * mask_expanded
                
                new_embeds_list.append(torch.cat([before, valid_patches, after], dim=0))
                
                if attention_mask is not None:
                    before_attn = attention_mask[b, :pos]
                    after_attn = attention_mask[b, pos:]
                    # Patches get attention mask based on patch_mask
                    if patch_mask is not None:
                        patch_attn = patch_mask[b].to(attention_mask.dtype)
                    else:
                        patch_attn = torch.ones(k_max, dtype=attention_mask.dtype, device=device)
                    new_attn_list.append(torch.cat([before_attn, patch_attn, after_attn], dim=0))
                
                if labels is not None:
                    before_labels = labels[b, :pos]
                    after_labels = labels[b, pos:]
                    # Patches get -100 (ignore in loss)
                    patch_labels = torch.full((k_max,), -100, dtype=labels.dtype, device=device)
                    new_labels_list.append(torch.cat([before_labels, patch_labels, after_labels], dim=0))
        
        # Stack all samples
        new_inputs_embeds = torch.stack(new_embeds_list, dim=0)  # [B, seq_len + k_max, hidden_dim]
        new_attention_mask = torch.stack(new_attn_list, dim=0) if new_attn_list is not None else None
        new_labels = torch.stack(new_labels_list, dim=0) if new_labels_list is not None else None
        
        return new_inputs_embeds, new_attention_mask, new_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        graph_data: Optional[Dict[str, Any]] = None,
        batch: Optional[torch.Tensor] = None,
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
            patch_positions: [B, 1] single position where patches should be inserted
            attention_mask: [B, seq_len] attention mask (used to identify non-padding tokens)
            patch_mask: [B, k_max] valid patch mask (optional)
            node_mask: [N] valid node mask (optional)
            labels: [B, seq_len] training labels (also used to identify instruction tokens: labels==-100)
            return_dict: whether to return dict
        
        Returns:
            CausalLMOutputWithPast
        
        Note:
            Instruction positions are computed dynamically from labels and attention_mask.
            - Training: instruction tokens are where labels == -100 (prompt tokens)
            - Inference: all non-padding tokens (attention_mask == 1) are instruction
        """
        # Get text embeddings
        
        inputs_embeds = self.get_llm_embeddings(input_ids)  # [B, seq_len, llm_hidden_dim]
        B, seq_len, _ = inputs_embeds.shape
        
        # If graph data provided, encode and fuse
        if graph_data is not None and batch is not None:
            # Extract instruction embeddings dynamically from labels and attention_mask
            # Training: instruction tokens are where labels == -100 (prompt) and attention_mask == 1 (not padding)
            # Inference: all tokens where attention_mask == 1 (entire prompt, no labels available)
            G = batch.max().item() + 1
            instr_emb_list = []
            
            for g in range(G):
                if g < B:
                    # Dynamically compute instruction positions from labels and attention_mask
                    if labels is not None:
                        # Training: instruction tokens are where labels == -100 (prompt tokens)
                        # and attention_mask == 1 (not padding)
                        is_instruction = (labels[g] == -100)
                        if attention_mask is not None:
                            is_instruction = is_instruction & (attention_mask[g] == 1)
                    else:
                        # Inference: all non-padding tokens are instruction (entire prompt)
                        # No labels available, so use all valid tokens
                        if attention_mask is not None:
                            is_instruction = (attention_mask[g] == 1)
                        else:
                            is_instruction = torch.ones(seq_len, dtype=torch.bool, device=inputs_embeds.device)
                    
                    # Extract instruction token embeddings
                    if is_instruction.any():
                        instr_tokens = inputs_embeds[g, is_instruction]  # [n_instr, llm_hidden_dim]
                        instr_emb_list.append(instr_tokens.mean(dim=0))
                    else:
                        instr_emb_list.append(torch.zeros(self.llm_hidden_dim, device=inputs_embeds.device))
                else:
                    instr_emb_list.append(torch.zeros(self.llm_hidden_dim, device=inputs_embeds.device))
            instr_emb = torch.stack(instr_emb_list, dim=0)  # [G, llm_hidden_dim]
            
            # Encode and patch
            
            # print("graph_data keys:", graph_data["value"].keys())
            # print("edge_feat_dist shape:", graph_data["value"]['edge_feat_dist'].shape)
            # print("edge_index shape:", graph_data["value"]['edge_index'].shape)

            # import ipdb; ipdb.set_trace()
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
                # Insert patches at specified positions (new behavior: INSERT, not replace)
                inputs_embeds, attention_mask, labels = self._inject_patches_into_sequence(
                    inputs_embeds=inputs_embeds[:G],  # Only process samples with graphs
                    patches=fused_patches,
                    patch_positions=patch_positions,
                    patch_mask=effective_patch_mask,
                    attention_mask=attention_mask[:G] if attention_mask is not None else None,
                    labels=labels[:G] if labels is not None else None,
                )
            else:
                # Fallback: Concatenate at beginning (old behavior for backward compatibility)
                inputs_embeds = torch.cat([fused_patches[:B], inputs_embeds], dim=1)
                
                # Adjust masks and labels
                k_max = self.config_octopus.patching.k_max
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
        patch_positions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
        return_patch_tokens_count: bool = False,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Generate text with graph context.
        
        Args:
            input_ids: [B, seq_len] token IDs
            graph_data: {'modality': str, 'value': {...}} graph input
            batch: [N] node-to-graph assignment
            patch_positions: [B, 1] single position where patches should be inserted
            attention_mask: [B, seq_len] attention mask (used to identify non-padding instruction tokens)
            patch_mask: [B, k_max] valid patch mask (optional)
            node_mask: [N] valid node mask (optional)
            return_patch_tokens_count: whether to return number of patch tokens added
            **generation_kwargs: additional generation parameters
        
        Returns:
            Generated token IDs (and optionally patch token count)
        
        Note:
            Instruction positions are computed dynamically from attention_mask.
            All non-padding tokens (attention_mask == 1) are treated as instruction.
        """
        self.eval()
        
        # Filter out internal tracking fields and non-generation kwargs
        generation_kwargs = {k: v for k, v in generation_kwargs.items() 
                           if not k.startswith("_") and k != "labels"}
        
        inputs_embeds = self.get_llm_embeddings(input_ids)
        B, seq_len, _ = inputs_embeds.shape
        
        if graph_data is not None and batch is not None:
            # Extract instruction embeddings dynamically from attention_mask
            # For generation (inference), all non-padding tokens are instruction (entire prompt)
            G = batch.max().item() + 1
            instr_emb_list = []
            
            for g in range(G):
                if g < B:
                    # Inference: all non-padding tokens are instruction (entire prompt)
                    if attention_mask is not None:
                        is_instruction = (attention_mask[g] == 1)
                    else:
                        is_instruction = torch.ones(seq_len, dtype=torch.bool, device=inputs_embeds.device)
                    
                    # Extract instruction token embeddings
                    if is_instruction.any():
                        instr_tokens = inputs_embeds[g, is_instruction]  # [n_instr, llm_hidden_dim]
                        instr_emb_list.append(instr_tokens.mean(dim=0))
                    else:
                        instr_emb_list.append(torch.zeros(self.llm_hidden_dim, device=inputs_embeds.device))
                else:
                    instr_emb_list.append(torch.zeros(self.llm_hidden_dim, device=inputs_embeds.device))
            instr_emb = torch.stack(instr_emb_list, dim=0)  # [G, llm_hidden_dim]
            
            patch_emb, out_patch_mask, node_emb = self.encode_and_patch(graph_data, instr_emb, batch)

            effective_patch_mask = out_patch_mask
            if patch_mask is not None:
                effective_patch_mask = effective_patch_mask & patch_mask[: effective_patch_mask.size(0)].bool()

            fused_patches = self.fuse_patches_with_nodes(patch_emb, node_emb, effective_patch_mask, node_mask, batch)
            
            if patch_positions is not None:
                # Insert patches at specified positions (new behavior: INSERT, not replace)
                inputs_embeds, attention_mask, _ = self._inject_patches_into_sequence(
                    inputs_embeds=inputs_embeds[:G],
                    patches=fused_patches,
                    patch_positions=patch_positions,
                    patch_mask=effective_patch_mask,
                    attention_mask=attention_mask[:G] if attention_mask is not None else None,
                    labels=None,  # No labels during generation
                )
            else:
                # Fallback: Concatenate at beginning
                k_max = self.config_octopus.patching.k_max
                inputs_embeds = torch.cat([fused_patches[:B], inputs_embeds], dim=1)
                if attention_mask is not None:
                    patch_attn_mask = torch.ones((B, k_max), dtype=attention_mask.dtype, device=attention_mask.device)
                    if effective_patch_mask is not None:
                        patch_attn_mask = effective_patch_mask[:B].to(attention_mask.dtype)
                    attention_mask = torch.cat([patch_attn_mask, attention_mask], dim=1)
        
        # Ensure max_new_tokens is set when using inputs_embeds
        if "max_new_tokens" not in generation_kwargs and "max_length" not in generation_kwargs:
            generation_kwargs["max_new_tokens"] = self.generation_config.max_new_tokens if hasattr(self, "generation_config") else 512
        
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        if return_patch_tokens_count:
            return (
                outputs,
                inputs_embeds.shape[1] - seq_len
            )        
        return outputs
    
    
    
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
        """
        DEPRECATED: This method is no longer needed.
        
        Multimodal components are trainable by default after model initialization.
        Use freeze_*() methods or apply_freezing_config() to selectively freeze components instead.
        """
        import warnings
        warnings.warn(
            "enable_multimodal_training() is deprecated and will be removed in a future version. "
            "Multimodal components are trainable by default. Use freeze_*() methods to freeze components.",
            DeprecationWarning,
            stacklevel=2
        )
        # Legacy behavior: ensure all multimodal components are trainable
        trainable_modules = [
            self.encoder,
            self.instr_norm,
            self.instr_proj,
            self.anchor_gate,
            self.edge_gate,
            self.patch_norm_pre,
            self.node_norm_pre,
            self.patch_proj,
            self.node_proj,
            self.fusion_blocks,
            self.output_proj,
            self.output_norm,
        ]
        
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
    
    def freeze_encoder(self):
        """Freeze the graph encoder."""
        self.encoder.freeze()
        print("Frozen encoder")
    
    def freeze_llm(self):
        """Freeze the LLM."""
        for param in self.llm_model.parameters():
            param.requires_grad = False
        print("Frozen LLM")
    
    def freeze_gates(self):
        """Freeze anchor and edge gates."""
        self.anchor_gate.freeze()
        self.edge_gate.freeze()
        print("Frozen gates (anchor_gate, edge_gate)")
    
    def freeze_fusion_blocks(self):
        """Freeze all fusion blocks."""
        for fusion_block in self.fusion_blocks:
            fusion_block.freeze()
        print("Frozen fusion blocks")
    
    def freeze_projections(self):
        """Freeze all projection layers and their associated normalization layers."""
        projection_modules = [
            self.instr_norm,
            self.instr_proj,
            self.patch_norm_pre,
            self.node_norm_pre,
            self.patch_proj,
            self.node_proj,
            self.output_proj,
            self.output_norm,
        ]
        for module in projection_modules:
            for param in module.parameters():
                param.requires_grad = False
        print("Frozen projection layers (instr_proj, patch_proj, node_proj, output_proj, output_norm)")
    
    def apply_freezing_config(self, freeze_config: Optional[Dict[str, bool]] = None):
        """
        Apply freezing configuration to model components.
        
        Args:
            freeze_config: Dictionary with freezing flags:
                - freeze_encoder: bool
                - freeze_llm: bool
                - freeze_gates: bool
                - freeze_fusion_blocks: bool
                - freeze_projections: bool
        
        If freeze_config is None, uses self.config_octopus freezing flags.
        
        This method should be called after applying PEFT to ensure proper freezing order.
        """
        if freeze_config is None:
            # Try to get from OctopusConfig if it has freezing attributes
            if hasattr(self.config_octopus, 'freeze_encoder'):
                freeze_config = {
                    'freeze_encoder': getattr(self.config_octopus, 'freeze_encoder', False),
                    'freeze_llm': getattr(self.config_octopus, 'freeze_llm', False),
                    'freeze_gates': getattr(self.config_octopus, 'freeze_gates', False),
                    'freeze_fusion_blocks': getattr(self.config_octopus, 'freeze_fusion_blocks', False),
                    'freeze_projections': getattr(self.config_octopus, 'freeze_projections', False),
                }
            else:
                # No freezing config available
                return
        
        # Apply freezing using individual freeze methods
        if freeze_config.get('freeze_encoder', False):
            self.freeze_encoder()
        
        if freeze_config.get('freeze_llm', False):
            self.freeze_llm()
        
        if freeze_config.get('freeze_gates', False):
            self.freeze_gates()
        
        if freeze_config.get('freeze_fusion_blocks', False):
            self.freeze_fusion_blocks()
        
        if freeze_config.get('freeze_projections', False):
            self.freeze_projections()