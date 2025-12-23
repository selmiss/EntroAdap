import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config, get_peft_config

from src.models.training_configs import GRPOConfig, SFTConfig, OctopusConfig as OctopusTrainingConfig
from src.models import Octopus
from src.models.octopus_config import (
    OctopusConfig,
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
    multimodal_config: OctopusTrainingConfig,
) -> Octopus:
    """
    Get the custom multi-modal model.
    
    IMPORTANT: 
    - If PEFT is enabled, it will be applied ONLY to the inner LLM model,
      not the entire Octopus wrapper. This allows the fusion blocks and 
      modality embeddings to be trained normally while using LoRA on the LLM.
    - Multimodal components are trainable by default after initialization.
    - Use the freezing options in multimodal_config to selectively freeze components:
      freeze_encoder, freeze_llm, freeze_gates, freeze_fusion_blocks, freeze_projections
    
    Args:
        model_args: Model configuration from TRL
        training_args: Training configuration
        multimodal_config: Multi-modal specific configuration (includes freezing options)
    
    Returns:
        Octopus instance with the specified configuration and freezing applied
    """
    # First, load the base LLM model
    # Handle both old (torch_dtype) and new (dtype) parameter names for backward compatibility
    dtype_value = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    torch_dtype = (
        dtype_value if dtype_value in ["auto", None] else getattr(torch, dtype_value)
    )
    quantization_config = get_quantization_config(model_args)
    
    # Default to "eager" attention implementation for Octopus compatibility
    # Octopus inherits from PreTrainedModel, which requires explicit attention implementation
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
    
    # Apply PEFT to the LLM model BEFORE wrapping it in Octopus
    # This ensures only the LLM uses LoRA, not the fusion blocks
    if model_args.use_peft:
        from peft import get_peft_model
        peft_config = get_peft_config(model_args)
        if peft_config is not None:
            llm_model = get_peft_model(llm_model, peft_config)
    
    # Build the config for the current graph-based Octopus model.
    #
    # NOTE: OctopusConfig historically described a different (token/embedding-based) multimodal model.
    # We map the overlapping "fusion" knobs onto OctopusConfig for backward compatibility so that
    # the training script can still configure the number of fusion blocks / heads / dims.
    enc_hidden = (
        multimodal_config.fusion_hidden_dim
        if multimodal_config.fusion_hidden_dim is not None
        else multimodal_config.modality_embedding_dim
    )
    mm_config = OctopusConfig(
        encoder=EncoderConfig(hidden_dim=int(enc_hidden)),
        patching=PatchingConfig(),  # use defaults; patching-specific knobs are not exposed in OctopusConfig
        fusion=FusionConfig(
            num_blocks=int(multimodal_config.num_fusion_blocks),
            num_heads=int(multimodal_config.num_attention_heads),
            hidden_dim=multimodal_config.fusion_hidden_dim,
            intermediate_dim=multimodal_config.fusion_intermediate_dim,
            dropout=float(multimodal_config.dropout),
        ),
    )

    # Create the multi-modal wrapper with the (possibly PEFT-wrapped) LLM
    multimodal_model = Octopus(llm_model=llm_model, config=mm_config)
    
    # Note: Multimodal components are trainable by default after initialization.
    # Apply freezing configuration if specified to selectively freeze components.
    freeze_config = {
        'freeze_encoder': getattr(multimodal_config, 'freeze_encoder', False),
        'freeze_llm': getattr(multimodal_config, 'freeze_llm', False),
        'freeze_gates': getattr(multimodal_config, 'freeze_gates', False),
        'freeze_fusion_blocks': getattr(multimodal_config, 'freeze_fusion_blocks', False),
        'freeze_projections': getattr(multimodal_config, 'freeze_projections', False),
    }
    multimodal_model.apply_freezing_config(freeze_config)
    
    return multimodal_model


def get_model_and_peft_config(
    model_args: ModelConfig,
    training_args: SFTConfig | GRPOConfig,
    multimodal_args: OctopusConfig | None = None,
):
    """
    Create the correct model (standard LLM vs custom multimodal) and return the PEFT config
    expected by TRL's trainers.

    - For **custom multimodal models**, LoRA/PEFT is applied inside `get_custom_model()` to the
      inner LLM only, so we must return `peft_config=None` to avoid double-wrapping by the trainer.
    - For **standard LLMs**, we return `peft_config=get_peft_config(model_args)` so the trainer
      applies PEFT in the standard TRL way.
    """
    if multimodal_args is not None and getattr(multimodal_args, "use_custom_model", False):
        model = get_custom_model(model_args, training_args, multimodal_args)
        peft_config = None
    else:
        model = get_model(model_args, training_args)
        peft_config = get_peft_config(model_args)
    return model, peft_config
