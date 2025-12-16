#!/usr/bin/env python3
# coding=utf-8
"""
Tests for multi-modal LLM model.
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from src.models.multimodal_llm import MultiModalLLM


class TestMultiModalLLM:
    """Tests for MultiModalLLM."""
    
    @pytest.fixture
    def small_llm(self):
        """Create a small LLM for testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2  # Use only 2 layers for speed
        config.n_head = 4
        config.n_embd = 256
        config.vocab_size = 1000
        
        llm = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
        return llm
    
    @pytest.fixture
    def multimodal_model(self, small_llm):
        """Create a multi-modal model for testing."""
        model = MultiModalLLM(
            llm_model=small_llm,
            modality_vocab_size=1000,
            modality_embedding_dim=256,
            num_fusion_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )
        return model
    
    def test_initialization(self, multimodal_model):
        """Test MultiModalLLM initialization."""
        assert isinstance(multimodal_model.modality_embedding, nn.Embedding)
        assert len(multimodal_model.fusion_blocks) == 2
        assert isinstance(multimodal_model.output_proj, nn.Linear)
        assert isinstance(multimodal_model.output_norm, nn.LayerNorm)
    
    def test_forward_text_only(self, multimodal_model):
        """Test forward pass with text only (no modality tokens)."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        outputs = multimodal_model(input_ids=input_ids)
        
        assert outputs.logits.shape == (batch_size, seq_len, 1000)
    
    def test_forward_with_modality_concatenation(self, multimodal_model):
        """Test forward pass with modality embeddings (concatenation mode)."""
        batch_size, text_seq_len, modality_seq_len = 2, 10, 8
        modality_dim = 256
        text_dim = 256  # LLM hidden dim
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        modality_embeddings = torch.randn(batch_size, modality_seq_len, modality_dim)
        cross_attn_text_embeddings = torch.randn(batch_size, 16, text_dim)
        
        outputs = multimodal_model(
            input_ids=input_ids,
            modality_embeddings=modality_embeddings,
            kv_embeddings=cross_attn_text_embeddings,
        )
        
        # Output should include both modality and text tokens
        expected_seq_len = modality_seq_len + text_seq_len
        assert outputs.logits.shape == (batch_size, expected_seq_len, 1000)
    
    def test_forward_with_attention_masks(self, multimodal_model):
        """Test forward pass with attention masks."""
        batch_size, text_seq_len, modality_seq_len = 2, 10, 8
        modality_dim = 256
        text_dim = 256
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        modality_embeddings = torch.randn(batch_size, modality_seq_len, modality_dim)
        cross_attn_text_embeddings = torch.randn(batch_size, 16, text_dim)
        
        # Create masks (1=valid, 0=padding)
        attention_mask = torch.ones(batch_size, text_seq_len, dtype=torch.long)
        modality_mask = torch.ones(batch_size, modality_seq_len, dtype=torch.long)
        text_mask = torch.ones(batch_size, 16, dtype=torch.long)
        
        # Set some padding
        attention_mask[:, -2:] = 0
        modality_mask[:, -2:] = 0
        text_mask[:, -4:] = 0
        
        outputs = multimodal_model(
            input_ids=input_ids,
            modality_embeddings=modality_embeddings,
            kv_embeddings=cross_attn_text_embeddings,
            attention_mask=attention_mask,
            modality_attention_mask=modality_mask,
            text_attention_mask=text_mask,
        )
        
        assert outputs.logits is not None
    
    def test_forward_with_labels(self, multimodal_model):
        """Test forward pass with labels for training."""
        batch_size, text_seq_len, modality_seq_len = 2, 10, 8
        modality_dim = 256
        text_dim = 256
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        modality_embeddings = torch.randn(batch_size, modality_seq_len, modality_dim)
        cross_attn_text_embeddings = torch.randn(batch_size, 16, text_dim)
        labels = torch.randint(0, 1000, (batch_size, text_seq_len))
        
        outputs = multimodal_model(
            input_ids=input_ids,
            modality_embeddings=modality_embeddings,
            kv_embeddings=cross_attn_text_embeddings,
            labels=labels,
        )
        
        assert outputs.loss is not None
        assert outputs.logits is not None
    
    def test_fuse_modality_and_text(self, multimodal_model):
        """Test fusion of modality and text embeddings."""
        batch_size, modality_seq_len, text_seq_len = 2, 8, 16
        modality_dim = 256
        text_dim = 256
        
        modality_embeddings = torch.randn(batch_size, modality_seq_len, modality_dim)
        cross_attn_text_embeddings = torch.randn(batch_size, text_seq_len, text_dim)
        
        fused_embeds = multimodal_model.fuse_modality_and_text(
            modality_embeddings=modality_embeddings,
            kv_embeddings=cross_attn_text_embeddings,
        )
        
        # Should output embeddings in LLM space
        assert fused_embeds.shape == (batch_size, modality_seq_len, multimodal_model.llm_hidden_dim)
    
    def test_generate(self, multimodal_model):
        """Test text generation with multimodal context."""
        batch_size, text_seq_len, modality_seq_len = 1, 5, 8
        num_new_tokens = 10
        modality_dim = 256
        text_dim = 256
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        modality_embeddings = torch.randn(batch_size, modality_seq_len, modality_dim)
        cross_attn_text_embeddings = torch.randn(batch_size, 16, text_dim)
        
        # Set to eval mode
        multimodal_model.eval()
        
        with torch.no_grad():
            generated_ids = multimodal_model.generate(
                input_ids=input_ids,
                modality_embeddings=modality_embeddings,
                kv_embeddings=cross_attn_text_embeddings,
                max_new_tokens=num_new_tokens,  # Generate new tokens beyond input
                num_beams=1,
                do_sample=False,
            )
        
        assert generated_ids.shape[0] == batch_size
        # When using inputs_embeds, generate() returns only the newly generated tokens
        assert generated_ids.shape[1] == num_new_tokens
    
    def test_position_based_injection(self, multimodal_model):
        """Test injecting modality embeddings at specific positions."""
        batch_size, text_seq_len, modality_seq_len = 2, 20, 3
        modality_dim = 256
        text_dim = 256
        
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        modality_embeddings = torch.randn(batch_size, modality_seq_len, modality_dim)
        cross_attn_text_embeddings = torch.randn(batch_size, 16, text_dim)
        
        # Specify positions to inject modality embeddings
        modality_positions = torch.tensor([[5, 6, 7], [10, 11, 12]], dtype=torch.long)
        
        outputs = multimodal_model(
            input_ids=input_ids,
            modality_embeddings=modality_embeddings,
            modality_positions=modality_positions,
            kv_embeddings=cross_attn_text_embeddings,
        )
        
        # Output length should be same as input (replacement, not concatenation)
        assert outputs.logits.shape == (batch_size, text_seq_len, 1000)
    
    def test_has_prepare_inputs_for_generation(self, multimodal_model):
        """Test that prepare_inputs_for_generation method exists and works."""
        # This method is required for generation, especially with PEFT
        assert hasattr(multimodal_model, 'prepare_inputs_for_generation')
        
        # Test that it can be called
        input_ids = torch.randint(0, 1000, (1, 5))
        result = multimodal_model.prepare_inputs_for_generation(input_ids)
        
        # Should return a dictionary with model inputs
        assert isinstance(result, dict)
        assert 'input_ids' in result or 'inputs_embeds' in result
    
    def test_has_can_generate(self, multimodal_model):
        """Test that can_generate method exists and returns True."""
        assert hasattr(multimodal_model, 'can_generate')
        assert callable(multimodal_model.can_generate)
        assert multimodal_model.can_generate() is True
    
    def test_has_reorder_cache(self, multimodal_model):
        """Test that _reorder_cache method exists."""
        assert hasattr(multimodal_model, '_reorder_cache')
        assert callable(multimodal_model._reorder_cache)
    
    @pytest.mark.slow
    def test_peft_compatibility(self, small_llm):
        """Test that the model works when PEFT is applied to the inner LLM."""
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            pytest.skip("PEFT not installed")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],  # GPT2 specific
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply PEFT to the LLM BEFORE wrapping (this is the correct approach)
        peft_llm = get_peft_model(small_llm, lora_config)
        
        # Now wrap the PEFT-enabled LLM in MultiModalLLM
        multimodal_model = MultiModalLLM(
            llm_model=peft_llm,
            modality_vocab_size=1000,
            modality_embedding_dim=256,
            num_fusion_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )
        
        # Test that required generation methods are still accessible
        assert hasattr(multimodal_model, 'prepare_inputs_for_generation')
        assert hasattr(multimodal_model, 'can_generate')
        assert hasattr(multimodal_model, '_reorder_cache')
        
        # Test forward pass with just text
        batch_size, text_seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, text_seq_len))
        
        outputs = multimodal_model(input_ids=input_ids)
        assert outputs.logits is not None
        
        # Test forward pass with multimodal inputs
        modality_embeddings = torch.randn(batch_size, 8, 256)
        cross_attn_text_embeddings = torch.randn(batch_size, 16, 256)
        
        outputs = multimodal_model(
            input_ids=input_ids,
            modality_embeddings=modality_embeddings,
            kv_embeddings=cross_attn_text_embeddings,
        )
        assert outputs.logits is not None
    
    def test_generation_methods_delegate_to_llm(self, multimodal_model, small_llm):
        """Test that generation methods properly delegate to the underlying LLM."""
        # Test prepare_inputs_for_generation
        input_ids = torch.randint(0, 1000, (1, 5))
        
        model_result = multimodal_model.prepare_inputs_for_generation(input_ids)
        llm_result = small_llm.prepare_inputs_for_generation(input_ids)
        
        # Both should return similar structure
        assert type(model_result) == type(llm_result)
        
        # Test can_generate
        assert multimodal_model.can_generate() == small_llm.can_generate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
