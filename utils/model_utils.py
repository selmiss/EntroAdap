import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config, get_peft_config

from src.configs import GRPOConfig, SFTConfig, MultiModalConfig
from src.models import MultiModalLLM
from src.models.multimodal_llm_config import (
    MultiModalLLMConfig,
    EncoderConfig,
    PatchingConfig,
    FusionConfig,
)


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
    """Get the model (standard LLM only)"""
    # Handle both old (torch_dtype) and new (dtype) parameter names for backward compatibility
    dtype_value = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    torch_dtype = (
        dtype_value if dtype_value in ["auto", None] else getattr(torch, dtype_value)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model


def get_custom_model(
    model_args: ModelConfig,
    training_args: SFTConfig | GRPOConfig,
    multimodal_config: MultiModalConfig,
) -> MultiModalLLM:
    """
    Get the custom multi-modal model.
    
    IMPORTANT: If PEFT is enabled, it will be applied ONLY to the inner LLM model,
    not the entire MultiModalLLM wrapper. This allows the fusion blocks and 
    modality embeddings to be trained normally while using LoRA on the LLM.
    
    Args:
        model_args: Model configuration from TRL
        training_args: Training configuration
        multimodal_config: Multi-modal specific configuration
    
    Returns:
        MultiModalLLM instance with the specified configuration
    """
    # First, load the base LLM model
    # Handle both old (torch_dtype) and new (dtype) parameter names for backward compatibility
    dtype_value = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    torch_dtype = (
        dtype_value if dtype_value in ["auto", None] else getattr(torch, dtype_value)
    )
    quantization_config = get_quantization_config(model_args)
    
    # Default to "eager" attention implementation for MultiModalLLM compatibility
    # MultiModalLLM inherits from PreTrainedModel, which requires explicit attention implementation
    attn_implementation = model_args.attn_implementation if model_args.attn_implementation is not None else "eager"
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    
    # Apply PEFT to the LLM model BEFORE wrapping it in MultiModalLLM
    # This ensures only the LLM uses LoRA, not the fusion blocks
    if model_args.use_peft:
        from peft import get_peft_model
        peft_config = get_peft_config(model_args)
        if peft_config is not None:
            llm_model = get_peft_model(llm_model, peft_config)
    
    # Build the config for the current graph-based MultiModalLLM.
    #
    # NOTE: MultiModalConfig historically described a different (token/embedding-based) multimodal model.
    # We map the overlapping "fusion" knobs onto MultiModalLLMConfig for backward compatibility so that
    # the training script can still configure the number of fusion blocks / heads / dims.
    enc_hidden = (
        multimodal_config.fusion_hidden_dim
        if multimodal_config.fusion_hidden_dim is not None
        else multimodal_config.modality_embedding_dim
    )
    mm_config = MultiModalLLMConfig(
        encoder=EncoderConfig(hidden_dim=int(enc_hidden)),
        patching=PatchingConfig(),  # use defaults; patching-specific knobs are not exposed in MultiModalConfig
        fusion=FusionConfig(
            num_blocks=int(multimodal_config.num_fusion_blocks),
            num_heads=int(multimodal_config.num_attention_heads),
            hidden_dim=multimodal_config.fusion_hidden_dim,
            intermediate_dim=multimodal_config.fusion_intermediate_dim,
            dropout=float(multimodal_config.dropout),
        ),
    )

    # Create the multi-modal wrapper with the (possibly PEFT-wrapped) LLM
    multimodal_model = MultiModalLLM(llm_model=llm_model, config=mm_config)
    
    # Ensure multimodal components are trainable (especially important when PEFT is used)
    multimodal_model.enable_multimodal_training()
    
    return multimodal_model
