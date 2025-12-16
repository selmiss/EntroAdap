#!/usr/bin/env python3
# coding=utf-8
"""
Integration tests for the complete multi-modal training pipeline.
"""

import pytest
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from trl import ModelConfig

from src.configs import SFTConfig, MultiModalConfig
from src.models import MultiModalLLM
from src.data_loader import MultiModalDataCollator
from utils.model_utils import get_custom_model


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        return tokenizer
    
    @pytest.fixture
    def small_config(self):
        """Create a small model config for fast testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 4
        config.n_embd = 256
        # Keep original vocab_size to match tokenizer
        return config
    
    @pytest.fixture
    def multimodal_model(self, small_config):
        """Create a small multimodal model for testing."""
        llm = AutoModelForCausalLM.from_config(small_config, attn_implementation="eager")
        model = MultiModalLLM(
            llm_model=llm,
            modality_vocab_size=1000,
            modality_embedding_dim=256,
            num_fusion_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )
        return model
    
    def test_end_to_end_forward_pass(self, multimodal_model, tokenizer):
        """Test complete forward pass from data to loss."""
        # Create sample data
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
        ]
        
        # Preprocess
        features = []
        for example in examples:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            tokenized = tokenizer(text, truncation=True, max_length=50)
            # Create random embeddings for multimodal data
            modality_dim = 256
            text_dim = 256  # Should match LLM hidden dim (256 for this small test model)
            features.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy(),
                "modality_embeddings": [[0.1] * modality_dim for _ in range(8)],
                "kv_embeddings": [[0.2] * text_dim for _ in range(16)],
            })
        
        # Collate
        collator = MultiModalDataCollator(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )
        batch = collator(features)
        
        # Forward pass
        multimodal_model.train()
        outputs = multimodal_model(**batch)
        
        # Check outputs
        assert outputs.loss is not None
        assert outputs.logits is not None
        assert not torch.isnan(outputs.loss)
        
        # Backward pass
        outputs.loss.backward()
    
    def test_end_to_end_generation(self, multimodal_model, tokenizer):
        """Test complete generation pipeline."""
        multimodal_model.eval()
        
        # Create input
        input_text = "What is"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        modality_dim = 256
        text_dim = 256  # Should match small test model's hidden dim (256)
        modality_embeddings = torch.randn(1, 8, modality_dim)
        cross_attn_text_embeddings = torch.randn(1, 16, text_dim)
        
        # Generate
        with torch.no_grad():
            generated_ids = multimodal_model.generate(
                input_ids=input_ids,
                modality_embeddings=modality_embeddings,
                kv_embeddings=cross_attn_text_embeddings,
                max_length=input_ids.shape[1] + 20,
                num_beams=1,
                do_sample=False,
            )
        
        # Check output
        assert generated_ids.shape[0] == 1
        assert generated_ids.shape[1] > input_ids.shape[1]
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        assert len(generated_text) > 0
    
    @pytest.mark.slow
    def test_model_loading_and_inference(self, tokenizer):
        """Test loading model through get_custom_model and running inference."""
        model_args = ModelConfig(
            model_name_or_path="gpt2",
            dtype="float32",
            attn_implementation="eager",
        )
        
        training_args = SFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            gradient_checkpointing=False,
        )
        
        multimodal_args = MultiModalConfig(
            use_custom_model=True,
            modality_vocab_size=1000,
            modality_embedding_dim=256,
            num_fusion_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )
        
        # Load model
        model = get_custom_model(model_args, training_args, multimodal_args)
        model.eval()
        
        # Create input
        input_ids = tokenizer.encode("Hello world", return_tensors="pt")
        modality_dim = 256
        # text_dim should match the LLM's hidden dimension (768 for GPT-2)
        text_dim = 768
        modality_embeddings = torch.randn(1, 8, modality_dim)
        cross_attn_text_embeddings = torch.randn(1, 16, text_dim)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                modality_embeddings=modality_embeddings,
                kv_embeddings=cross_attn_text_embeddings,
            )
        
        assert outputs.logits is not None
        assert not torch.isnan(outputs.logits).any()
    
    def test_batch_processing_different_lengths(self, multimodal_model, tokenizer):
        """Test processing batches with varying sequence lengths."""
        # Create examples with different lengths
        modality_dim = 256
        text_dim = 256  # Should match small test model's hidden dim (256)
        features = [
            {
                "input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "attention_mask": [1] * 10,
                "labels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "modality_embeddings": [[0.1] * modality_dim for _ in range(12)],
                "kv_embeddings": [[0.2] * text_dim for _ in range(20)],
            },
            {
                "input_ids": [11, 12, 13],
                "attention_mask": [1] * 3,
                "labels": [11, 12, 13],
                "modality_embeddings": [[0.3] * modality_dim for _ in range(5)],
                "kv_embeddings": [[0.4] * text_dim for _ in range(8)],
            },
            {
                "input_ids": [14, 15, 16, 17, 18, 19, 20],
                "attention_mask": [1] * 7,
                "labels": [14, 15, 16, 17, 18, 19, 20],
                "modality_embeddings": [[0.5] * modality_dim for _ in range(8)],
                "kv_embeddings": [[0.6] * text_dim for _ in range(15)],
            },
        ]
        
        # Collate
        collator = MultiModalDataCollator(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )
        batch = collator(features)
        
        # Check batch shapes
        batch_size = 3
        assert batch["input_ids"].shape[0] == batch_size
        assert batch["modality_embeddings"].shape[0] == batch_size
        assert batch["kv_embeddings"].shape[0] == batch_size
        
        # All sequences should be padded to same length
        assert batch["input_ids"].shape[1] == 10  # Max text length
        assert batch["modality_embeddings"].shape[1] == 12  # Max modality length
        assert batch["kv_embeddings"].shape[1] == 20  # Max cross-attn length
        
        # Forward pass
        multimodal_model.train()
        outputs = multimodal_model(**batch)
        
        assert outputs.loss is not None
        assert not torch.isnan(outputs.loss)


class TestConfigurationIntegration:
    """Test configuration parsing and usage."""
    
    def test_multimodal_config_in_training(self):
        """Test that MultiModalConfig integrates properly with training args."""
        multimodal_args = MultiModalConfig(
            use_custom_model=True,
            modality_vocab_size=10000,
            modality_embedding_dim=768,
            num_fusion_blocks=4,
            num_attention_heads=8,
        )
        
        training_args = SFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=2,
        )
        
        # Check that configs can coexist
        assert multimodal_args.use_custom_model
        assert training_args.per_device_train_batch_size == 2
    
    def test_multimodal_disabled(self):
        """Test that multimodal can be disabled."""
        multimodal_args = MultiModalConfig(
            use_custom_model=False,
        )
        
        assert not multimodal_args.use_custom_model


class TestPEFTIntegration:
    """Test PEFT/LoRA integration with multimodal models."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small model config for fast testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 4
        config.n_embd = 256
        return config
    
    @pytest.fixture
    def multimodal_model(self, small_config):
        """Create a small multimodal model for testing."""
        llm = AutoModelForCausalLM.from_config(small_config, attn_implementation="eager")
        model = MultiModalLLM(
            llm_model=llm,
            modality_vocab_size=1000,
            modality_embedding_dim=256,
            num_fusion_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )
        return model
    
    @pytest.mark.slow
    def test_peft_wrapped_model_has_generation_methods(self, small_config):
        """Test that models with PEFT on inner LLM have all required generation methods."""
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            pytest.skip("PEFT not installed")
        
        # Create LLM
        llm = AutoModelForCausalLM.from_config(small_config, attn_implementation="eager")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],  # GPT2 specific
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply PEFT to LLM before wrapping
        peft_llm = get_peft_model(llm, lora_config)
        
        # Wrap in MultiModalLLM
        multimodal_model = MultiModalLLM(
            llm_model=peft_llm,
            modality_vocab_size=1000,
            modality_embedding_dim=256,
            num_fusion_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )
        
        # Test that all required methods exist and are callable
        required_methods = [
            'prepare_inputs_for_generation',
            'can_generate',
            '_reorder_cache',
            'generate',
            'forward',
        ]
        
        for method_name in required_methods:
            assert hasattr(multimodal_model, method_name), f"Model missing {method_name}"
            method = getattr(multimodal_model, method_name)
            assert callable(method), f"{method_name} is not callable"
    
    @pytest.mark.slow
    def test_peft_model_training_step(self, small_config):
        """Test that model with PEFT on inner LLM can perform a training step."""
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            pytest.skip("PEFT not installed")
        
        # Create LLM
        llm = AutoModelForCausalLM.from_config(small_config, attn_implementation="eager")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply PEFT to LLM before wrapping
        peft_llm = get_peft_model(llm, lora_config)
        
        # Wrap in MultiModalLLM
        multimodal_model = MultiModalLLM(
            llm_model=peft_llm,
            modality_vocab_size=1000,
            modality_embedding_dim=256,
            num_fusion_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )
        multimodal_model.train()
        
        # Create sample batch
        batch_size, seq_len = 2, 10
        modality_dim = 256
        text_dim = 256  # Should match small test model's hidden dim (256)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = input_ids.clone()
        modality_embeddings = torch.randn(batch_size, 8, modality_dim)
        cross_attn_text_embeddings = torch.randn(batch_size, 16, text_dim)
        
        # Forward pass
        outputs = multimodal_model(
            input_ids=input_ids,
            modality_embeddings=modality_embeddings,
            kv_embeddings=cross_attn_text_embeddings,
            labels=labels,
        )
        
        # Check outputs
        assert outputs.loss is not None
        assert not torch.isnan(outputs.loss)
        
        # Backward pass
        outputs.loss.backward()
    
    @pytest.mark.slow
    def test_get_custom_model_with_peft(self):
        """Test loading custom model through get_custom_model with PEFT config."""
        try:
            from peft import LoraConfig
        except ImportError:
            pytest.skip("PEFT not installed")
        
        # Use a smaller model for faster testing
        model_args = ModelConfig(
            model_name_or_path="gpt2",
            dtype="float32",
            attn_implementation="eager",
            use_peft=True,
            lora_r=8,
            lora_alpha=16,
            lora_target_modules=["c_attn", "c_proj"],  # GPT2 specific modules
        )
        
        training_args = SFTConfig(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            gradient_checkpointing=False,
        )
        
        multimodal_args = MultiModalConfig(
            use_custom_model=True,
            modality_vocab_size=1000,
            modality_embedding_dim=768,  # Match GPT2's hidden size
            num_fusion_blocks=2,
            num_attention_heads=4,
            dropout=0.1,
        )
        
        # Load model with PEFT (PEFT is applied to inner LLM inside get_custom_model)
        model = get_custom_model(model_args, training_args, multimodal_args)
        
        # Should have PEFT methods accessible
        assert hasattr(model, 'prepare_inputs_for_generation')
        assert hasattr(model, 'can_generate')
        
        # The inner llm_model should be a PEFT model
        assert hasattr(model.llm_model, 'peft_config')
        
        # Test that we can do a forward pass
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, model.llm_model.config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids=input_ids)
        assert outputs.logits is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
