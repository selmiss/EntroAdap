import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .fusion_blocks import FusionBlock


class MultiModalLLM(PreTrainedModel):
    """Multi-modal Language Model that combines modality embeddings with text through fusion blocks before feeding into a standard LLM."""
    
    def __init__(
        self,
        llm_model: AutoModelForCausalLM,
        modality_vocab_size: int,
        modality_embedding_dim: int,
        num_fusion_blocks: int = 4,
        num_attention_heads: int = 8,
        fusion_hidden_dim: Optional[int] = None,
        fusion_intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Initialize the MultiModalLLM with the LLM's config."""
        # Initialize with the LLM's config
        super().__init__(llm_model.config)
        
        self.llm_model = llm_model
        self.llm_hidden_dim = llm_model.config.hidden_size
        
        # Modality embedding layer
        self.modality_embedding = nn.Embedding(modality_vocab_size, modality_embedding_dim)
        
        # Fusion blocks configuration
        if fusion_hidden_dim is None:
            fusion_hidden_dim = modality_embedding_dim
        
        # Initial projection to fusion hidden dimension if needed
        if modality_embedding_dim != fusion_hidden_dim:
            self.modality_proj = nn.Linear(modality_embedding_dim, fusion_hidden_dim)
        else:
            self.modality_proj = nn.Identity()
        
        # Text embedding projection to match fusion hidden dimension
        if self.llm_hidden_dim != fusion_hidden_dim:
            self.text_embedding_proj = nn.Linear(self.llm_hidden_dim, fusion_hidden_dim)
        else:
            self.text_embedding_proj = nn.Identity()
        
        # Fusion blocks
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(
                hidden_dim=fusion_hidden_dim,
                num_heads=num_attention_heads,
                intermediate_dim=fusion_intermediate_dim,
                dropout=dropout,
            )
            for _ in range(num_fusion_blocks)
        ])
        
        # Final projection to LLM embedding space
        self.output_proj = nn.Linear(fusion_hidden_dim, self.llm_hidden_dim)
        
        # Layer norm before combining with LLM
        self.output_norm = nn.LayerNorm(self.llm_hidden_dim)
        
    def get_llm_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings from the LLM's embedding layer."""
        embeds = self.llm_model.get_input_embeddings()(input_ids)
        # Ensure embeddings require grad even if the embedding layer is frozen (e.g., with PEFT)
        # This is necessary for gradient flow through the fusion blocks
        if not embeds.requires_grad and self.training:
            embeds.requires_grad_(True)
        return embeds
    
    def fuse_modality_and_text(
        self,
        modality_embeddings: torch.Tensor,
        kv_embeddings: torch.Tensor,
        modality_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse modality embeddings with text embeddings through fusion blocks.
        
        Args:
            modality_embeddings: Pre-computed modality embeddings (batch, modality_seq_len, modality_dim)
            kv_embeddings: Pre-computed text embeddings (batch, text_seq_len, text_dim)
            modality_attention_mask: Attention mask for modality (1=valid, 0=padding)
            text_attention_mask: Attention mask for text (1=valid, 0=padding)
        """
        # Project modality embeddings to fusion hidden dimension
        modality_embeds = self.modality_proj(modality_embeddings)  # (batch, modality_seq_len, fusion_hidden_dim)
        
        # Project text embeddings to fusion hidden dimension
        text_embeds = self.text_embedding_proj(kv_embeddings)  # (batch, text_seq_len, fusion_hidden_dim)
        
        # Prepare key_padding_mask for PyTorch's MultiheadAttention
        # PyTorch expects boolean masks where True = padding (ignore), False = valid (attend)
        # Input masks are typically 1=valid, 0=padding, so we need to invert them
        key_padding_mask = None
        if modality_attention_mask is not None:
            # Convert from (1=valid, 0=padding) to (True=padding, False=valid)
            key_padding_mask = (modality_attention_mask == 0)
        
        cross_key_padding_mask = None
        if text_attention_mask is not None:
            # Convert from (1=valid, 0=padding) to (True=padding, False=valid)
            cross_key_padding_mask = (text_attention_mask == 0)
        
        # Apply fusion blocks
        hidden_states = modality_embeds
        for fusion_block in self.fusion_blocks:
            hidden_states = fusion_block(
                hidden_states=hidden_states,
                key_value_states=text_embeds,
                key_padding_mask=key_padding_mask,
                cross_key_padding_mask=cross_key_padding_mask,
            )
        
        # Project to LLM embedding space
        fused_embeds = self.output_proj(hidden_states)  # (batch, modality_seq_len, llm_hidden_dim)
        fused_embeds = self.output_norm(fused_embeds)
        
        return fused_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        modality_embeddings: Optional[torch.Tensor] = None,
        modality_positions: Optional[torch.Tensor] = None,
        kv_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        modality_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass through the multi-modal LLM.
        
        Args:
            input_ids: Token IDs for the main text sequence
            modality_embeddings: Pre-computed modality embeddings (batch, modality_seq_len, modality_dim)
            modality_positions: Positions to insert modality embeddings
            kv_embeddings: Pre-computed text embeddings for cross-attention (batch, text_seq_len, text_dim)
            attention_mask: Attention mask for main text
            modality_attention_mask: Attention mask for modality
            text_attention_mask: Attention mask for cross-attention text
            labels: Labels for training
            return_dict: Whether to return a dictionary
        """

        
        # Get input embeddings
        inputs_embeds = self.get_llm_embeddings(input_ids)
        
        # If modality embeddings are provided, fuse them and insert into the sequence
        if modality_embeddings is not None and kv_embeddings is not None:
            # Fuse modality with text
            fused_embeds = self.fuse_modality_and_text(
                modality_embeddings=modality_embeddings,
                kv_embeddings=kv_embeddings,
                modality_attention_mask=modality_attention_mask,
                text_attention_mask=text_attention_mask,
            )
            
            # Insert fused embeddings at specified positions
            if modality_positions is not None:
                batch_size, seq_len, hidden_dim = inputs_embeds.shape
                
                # Create output embeddings
                for batch_idx in range(batch_size):
                    for pos_idx, insert_pos in enumerate(modality_positions[batch_idx]):
                        if insert_pos >= 0 and insert_pos < seq_len:
                            inputs_embeds[batch_idx, insert_pos] = fused_embeds[batch_idx, pos_idx]
            else:
                # If no positions specified, concatenate at the beginning
                inputs_embeds = torch.cat([fused_embeds, inputs_embeds], dim=1)
                
                # Adjust attention mask and labels accordingly
                if attention_mask is not None:
                    modality_mask = torch.ones(
                        (attention_mask.shape[0], fused_embeds.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([modality_mask, attention_mask], dim=1)
                
                if labels is not None:
                    # Pad labels with -100 (ignore index) for modality tokens
                    ignore_labels = torch.full(
                        (labels.shape[0], fused_embeds.shape[1]),
                        fill_value=-100,
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    labels = torch.cat([ignore_labels, labels], dim=1)
        
        # Forward through LLM
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs,
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        modality_embeddings: Optional[torch.Tensor] = None,
        modality_positions: Optional[torch.Tensor] = None,
        kv_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        modality_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate text using the multi-modal LLM.
        
        Args:
            input_ids: Token IDs for the main text sequence
            modality_embeddings: Pre-computed modality embeddings
            modality_positions: Positions to insert modality embeddings
            kv_embeddings: Pre-computed text embeddings for cross-attention
            attention_mask: Attention mask for main text
            modality_attention_mask: Attention mask for modality
            text_attention_mask: Attention mask for cross-attention text
        """
        # Get input embeddings with fused modality
        inputs_embeds = self.get_llm_embeddings(input_ids)
        
        if modality_embeddings is not None and kv_embeddings is not None:
            fused_embeds = self.fuse_modality_and_text(
                modality_embeddings=modality_embeddings,
                kv_embeddings=kv_embeddings,
                modality_attention_mask=modality_attention_mask,
                text_attention_mask=text_attention_mask,
            )
            
            if modality_positions is not None:
                batch_size, seq_len, hidden_dim = inputs_embeds.shape
                for batch_idx in range(batch_size):
                    for pos_idx, insert_pos in enumerate(modality_positions[batch_idx]):
                        if insert_pos >= 0 and insert_pos < seq_len:
                            inputs_embeds[batch_idx, insert_pos] = fused_embeds[batch_idx, pos_idx]
            else:
                inputs_embeds = torch.cat([fused_embeds, inputs_embeds], dim=1)
                if attention_mask is not None:
                    modality_mask = torch.ones(
                        (attention_mask.shape[0], fused_embeds.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([modality_mask, attention_mask], dim=1)
        
        # Generate using LLM
        return self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation - delegates to the underlying LLM model."""
        return self.llm_model.prepare_inputs_for_generation(input_ids, **kwargs)
    
    def can_generate(self):
        """Check if the model can generate - delegates to the underlying LLM model."""
        # During __init__, llm_model might not be set yet, so return True by default
        if hasattr(self, 'llm_model'):
            return self.llm_model.can_generate()
        return True
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search - delegates to the underlying LLM model."""
        return self.llm_model._reorder_cache(past_key_values, beam_idx)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing - delegates to the underlying LLM model."""
        if hasattr(self.llm_model, 'gradient_checkpointing_enable'):
            self.llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing - delegates to the underlying LLM model."""
        if hasattr(self.llm_model, 'gradient_checkpointing_disable'):
            self.llm_model.gradient_checkpointing_disable()  
        else:
            super().gradient_checkpointing_disable()
    
    def enable_multimodal_training(self):
        """
        Ensure multimodal components are trainable.
        Call this after applying PEFT to the inner LLM to ensure fusion blocks and embeddings remain trainable.
        """
        # Ensure fusion components require gradients
        for param in self.modality_embedding.parameters():
            param.requires_grad = True
        
        for param in self.modality_proj.parameters():
            param.requires_grad = True
        
        for param in self.text_embedding_proj.parameters():
            param.requires_grad = True
        
        for param in self.fusion_blocks.parameters():
            param.requires_grad = True
        
        for param in self.output_proj.parameters():
            param.requires_grad = True
        
        for param in self.output_norm.parameters():
            param.requires_grad = True
