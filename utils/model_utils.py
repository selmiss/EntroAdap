import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from safetensors.torch import load_file
import json
from trl import ModelConfig, get_kbit_device_map, get_quantization_config, get_peft_config
from transformers import AutoConfig
from src.models.training_configs import GRPOConfig, SFTConfig, OctopusConfig as OctopusTrainingConfig
from src.models import Octopus
from src.models.octopus_config import (
    OctopusConfig,
    EncoderConfig,
    PatchingConfig,
    FusionConfig,
    PredictionHeadConfig,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig, multimodal_args=None) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer_path = model_args.model_name_or_path
    if tokenizer_path is None and multimodal_args is not None:
        tokenizer_path = getattr(multimodal_args, 'prepared_checkpoint_path', None) or getattr(multimodal_args, 'octopus_checkpoint_path', None)
    
    if tokenizer_path is None:
        raise ValueError("Either model_name_or_path or a checkpoint path must be provided")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        fix_mistral_regex=True,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def load_encoder_weights_from_checkpoint(encoder, checkpoint_path: str):
    """
    Load pretrained encoder weights from a checkpoint directory.
    
    Args:
        encoder: The AAEncoder instance to load weights into
        checkpoint_path: Path to checkpoint directory containing model.safetensors or pytorch_model.bin
    
    Returns:
        encoder: The encoder with loaded weights
    """
    if not os.path.isdir(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist or is not a directory: {checkpoint_path}")
    
    # Try loading safetensors first (preferred), then fall back to pytorch_model.bin
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        print(f"  → Loading from: {safetensors_path}")
        logger.info(f"Loading encoder weights from {safetensors_path}")
        state_dict = load_file(safetensors_path)
    elif os.path.exists(pytorch_path):
        print(f"  → Loading from: {pytorch_path}")
        logger.info(f"Loading encoder weights from {pytorch_path}")
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise ValueError(
            f"No model weights found in {checkpoint_path}. "
            f"Expected either 'model.safetensors' or 'pytorch_model.bin'"
        )
    
    print(f"  → Total keys in checkpoint: {len(state_dict)}")
    
    # Filter only encoder weights (ignore other components if checkpoint contains full model)
    encoder_state_dict = {}
    for key, value in state_dict.items():
        # Handle both direct encoder checkpoints and full model checkpoints
        if key.startswith("encoder."):
            # Remove "encoder." prefix if present
            new_key = key[len("encoder."):]
            encoder_state_dict[new_key] = value
        elif not any(key.startswith(prefix) for prefix in ["llm_model.", "instr_proj.", "fusion_blocks.", "gates.", "patching."]):
            # If it doesn't start with other known prefixes, assume it's an encoder weight
            encoder_state_dict[key] = value
    
    if not encoder_state_dict:
        logger.warning(f"No encoder weights found in checkpoint {checkpoint_path}. Loading all weights into encoder.")
        encoder_state_dict = state_dict
    
    print(f"  → Encoder-specific keys: {len(encoder_state_dict)}")
    
    # Load the weights into the encoder
    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
    
    if missing_keys:
        print(f"  ⚠ Missing keys: {len(missing_keys)} (these will be randomly initialized)")
        logger.warning(f"Missing keys when loading encoder: {missing_keys}")
    if unexpected_keys:
        print(f"  ℹ Unexpected keys: {len(unexpected_keys)} (ignored, e.g., pretraining heads)")
        logger.warning(f"Unexpected keys when loading encoder: {unexpected_keys}")
    
    loaded_keys = len(encoder_state_dict) - len(missing_keys)
    print(f"  → Successfully loaded: {loaded_keys} parameter groups")
    
    logger.info(f"Successfully loaded encoder weights from {checkpoint_path}")
    return encoder


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

def load_prepared_octopus_from_checkpoint(
    checkpoint_path: str,
    model_args,
    multimodal_config = None,
) -> Octopus:
    """
    Load a prepared Octopus model from a checkpoint directory.
    """
    if multimodal_config is None:
        multimodal_config = model_args
    
    dtype_value = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    torch_dtype = (
        dtype_value if dtype_value in ["auto", None] else getattr(torch, dtype_value)
    )
    quantization_config = get_quantization_config(model_args)
    
    attn_implementation = getattr(model_args, "attn_implementation", "eager")
    
    config_kwargs = dict(
        revision=getattr(model_args, "model_revision", None),
        trust_remote_code=getattr(model_args, "trust_remote_code", False),
    )
    
    # Kwargs for creating model from config (from_config only accepts these)
    model_init_kwargs = dict(
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    
    # Add quantization config if needed (from_config does accept this)
    if quantization_config is not None:
        model_init_kwargs['quantization_config'] = quantization_config
        model_init_kwargs['device_map'] = get_kbit_device_map()
    
    # Load config from checkpoint, then create model
    # Weights will be loaded later via load_state_dict  
    logger.info(f"Loading LLM config from checkpoint: {checkpoint_path}")
    llm_config = AutoConfig.from_pretrained(checkpoint_path, **config_kwargs)
    logger.info(f"Creating LLM model structure from config")
    
    # Use transformers' _fast_init to skip weight initialization (fast and safe)
    # This uses meta device internally but handles buffer materialization properly
    if quantization_config is None:
        # Use _fast_init context to skip initialization
        from transformers.modeling_utils import no_init_weights
        with no_init_weights():
            llm_model = AutoModelForCausalLM.from_config(config=llm_config, **model_init_kwargs)
    else:
        # Quantization requires normal initialization
        llm_model = AutoModelForCausalLM.from_config(config=llm_config, **model_init_kwargs)
    encoder_dim = (
        multimodal_config.encoder_hidden_dim
        if multimodal_config.encoder_hidden_dim is not None
        else multimodal_config.modality_embedding_dim
    )
    
    fusion_dim = (
        multimodal_config.fusion_hidden_dim
        if multimodal_config.fusion_hidden_dim is not None
        else multimodal_config.modality_embedding_dim
    )
    
    mm_config = OctopusConfig(
        encoder=EncoderConfig(
            hidden_dim=encoder_dim,
            num_layers=6,
            dropout=multimodal_config.dropout,
        ),
        patching=PatchingConfig(),
        fusion=FusionConfig(
            num_blocks=multimodal_config.num_fusion_blocks,
            num_heads=multimodal_config.num_attention_heads,
            hidden_dim=fusion_dim,
            intermediate_dim=multimodal_config.fusion_intermediate_dim,
            dropout=multimodal_config.dropout,
        ),
        prediction_head=PredictionHeadConfig(
            task_type=getattr(multimodal_config, 'task_type', None),
            num_labels=getattr(multimodal_config, 'num_labels', 2),
            pooling_strategy=getattr(multimodal_config, 'pooling_strategy', 'last'),
            hidden_dim=getattr(multimodal_config, 'head_hidden_dim', None),
            dropout=getattr(multimodal_config, 'head_dropout', 0.1),
            use_dual_loss=getattr(multimodal_config, 'use_dual_loss', False),
            lm_loss_weight=getattr(multimodal_config, 'lm_loss_weight', 0.5),
        ),
    )
    
    # Create Octopus model (this initializes the architecture)
    logger.info("Creating Octopus architecture")
    
    # Log prediction head config if enabled
    if mm_config.prediction_head.task_type is not None:
        logger.info(f"Configuring prediction head: task_type={mm_config.prediction_head.task_type}, "
                   f"use_dual_loss={mm_config.prediction_head.use_dual_loss}, "
                   f"lm_loss_weight={mm_config.prediction_head.lm_loss_weight}")
    
    multimodal_model = Octopus(llm_model=llm_model, config=mm_config)

    if getattr(model_args, "use_peft", False):
        from peft import get_peft_model
        logger.info("Applying LoRA wrapper to LLM")
        peft_config = get_peft_config(model_args)
        if peft_config is not None:
            multimodal_model.llm_model = get_peft_model(multimodal_model.llm_model, peft_config)

    
    logger.info(f"Loading Octopus weights from {checkpoint_path}")
    
    # Check if there's a model.safetensors.index.json (sharded) or model.safetensors (single file)
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    single_file_path = os.path.join(checkpoint_path, "model.safetensors")
    
    if os.path.exists(index_path):
        # Load from sharded files
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index.get("weight_map", {})
        state_dict = {}
        loaded_files = set()
        
        for param_name, shard_file in tqdm(weight_map.items(), desc="Loading shards"):
            if shard_file not in loaded_files:
                shard_path = os.path.join(checkpoint_path, shard_file)
                # logger.info(f"Loading shard: {shard_file}")
                shard_state = load_file(shard_path)
                state_dict.update(shard_state)
                loaded_files.add(shard_file)
        
        logger.info(f"Loaded {len(state_dict)} parameters from {len(loaded_files)} shards")
    
        
    elif os.path.exists(single_file_path):
        # Load from single file
        logger.info(f"Loading from single file: {single_file_path}")
        state_dict = load_file(single_file_path)
    else:
        raise FileNotFoundError(
            f"Could not find model weights at {checkpoint_path}. "
            f"Expected either {index_path} or {single_file_path}"
        )
    
    # Load the state dict into the model
    missing_keys, unexpected_keys = multimodal_model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {missing_keys[:10]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys[:10]}...")
    if not missing_keys and not unexpected_keys:
        logger.info("All keys loaded successfully")
    
    logger.info("Prepared Octopus model loaded successfully from checkpoint")
    
    # Apply freezing configuration
    freeze_config = {
        'freeze_encoder': getattr(multimodal_config, 'freeze_encoder', False),
        'freeze_llm': getattr(multimodal_config, 'freeze_llm', False),
        'freeze_gates': getattr(multimodal_config, 'freeze_gates', False),
        'freeze_fusion_blocks': getattr(multimodal_config, 'freeze_fusion_blocks', False),
        'freeze_projections': getattr(multimodal_config, 'freeze_projections', False),
    }
    multimodal_model.apply_freezing_config(freeze_config)

    return multimodal_model


def _load_octopus_from_checkpoint(
    checkpoint_path: str,
    model_args: ModelConfig,
    training_args: SFTConfig | GRPOConfig,
    multimodal_config: OctopusTrainingConfig,
) -> Octopus:
    """
    Load a trained Octopus model from a checkpoint directory with sharded weights.
    
    This function:
    1. Loads the base LLM from the checkpoint (without LoRA wrapper)
    2. Creates the Octopus wrapper with the same architecture
    3. Loads all Octopus weights (encoder, fusion blocks, gates, projections)
    4. Applies LoRA wrapper if needed (for continued training with LoRA)
    
    Args:
        checkpoint_path: Path to checkpoint directory with model-*.safetensors files
        model_args: Model configuration
        training_args: Training configuration
        multimodal_config: Multi-modal configuration
        
    Returns:
        Octopus model with loaded weights
    """

    
    # Load LLM from checkpoint WITHOUT LoRA wrapper first
    # This ensures key names match the saved checkpoint
    dtype_value = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    torch_dtype = (
        dtype_value if dtype_value in ["auto", None] else getattr(torch, dtype_value)
    )
    quantization_config = get_quantization_config(model_args)
    
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
    
    # Load LLM from the checkpoint directory (not from HF hub)
    logger.info(f"Loading LLM from checkpoint: {checkpoint_path}")
    llm_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        **model_kwargs,
    )
    
    # Build Octopus config
    encoder_dim = (
        multimodal_config.encoder_hidden_dim
        if multimodal_config.encoder_hidden_dim is not None
        else multimodal_config.modality_embedding_dim
    )
    
    fusion_dim = (
        multimodal_config.fusion_hidden_dim
        if multimodal_config.fusion_hidden_dim is not None
        else multimodal_config.modality_embedding_dim
    )
    
    mm_config = OctopusConfig(
        encoder=EncoderConfig(
            hidden_dim=encoder_dim,
            num_layers=6,
            dropout=multimodal_config.dropout,
        ),
        patching=PatchingConfig(
        ),
        fusion=FusionConfig(
            num_blocks=multimodal_config.num_fusion_blocks,
            num_heads=multimodal_config.num_attention_heads,
            hidden_dim=fusion_dim,
            intermediate_dim=multimodal_config.fusion_intermediate_dim,
            dropout=multimodal_config.dropout,
        ),
        prediction_head=PredictionHeadConfig(
            task_type=getattr(multimodal_config, 'task_type', None),
            num_labels=getattr(multimodal_config, 'num_labels', 2),
            pooling_strategy=getattr(multimodal_config, 'pooling_strategy', 'last'),
            hidden_dim=getattr(multimodal_config, 'head_hidden_dim', None),
            dropout=getattr(multimodal_config, 'head_dropout', 0.1),
            use_dual_loss=getattr(multimodal_config, 'use_dual_loss', False),
            lm_loss_weight=getattr(multimodal_config, 'lm_loss_weight', 0.5),
        ),
    )
    
    # Create Octopus model (this initializes the architecture)
    logger.info("Creating Octopus architecture")
    
    # Log prediction head config if enabled
    if mm_config.prediction_head.task_type is not None:
        logger.info(f"Configuring prediction head: task_type={mm_config.prediction_head.task_type}, "
                   f"use_dual_loss={mm_config.prediction_head.use_dual_loss}, "
                   f"lm_loss_weight={mm_config.prediction_head.lm_loss_weight}")
    
    multimodal_model = Octopus(llm_model=llm_model, config=mm_config)
    
    # Load Octopus-specific weights from checkpoint
    logger.info(f"Loading Octopus weights from {checkpoint_path}")
    
    # Check if there's a model.safetensors.index.json (sharded) or model.safetensors (single file)
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    single_file_path = os.path.join(checkpoint_path, "model.safetensors")
    
    if os.path.exists(index_path):
        # Load from sharded files
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index.get("weight_map", {})
        state_dict = {}
        loaded_files = set()
        
        for param_name, shard_file in weight_map.items():
            if shard_file not in loaded_files:
                shard_path = os.path.join(checkpoint_path, shard_file)
                logger.info(f"Loading shard: {shard_file}")
                shard_state = load_file(shard_path)
                state_dict.update(shard_state)
                loaded_files.add(shard_file)
        
        logger.info(f"Loaded {len(state_dict)} parameters from {len(loaded_files)} shards")
        
    elif os.path.exists(single_file_path):
        # Load from single file
        logger.info(f"Loading from single file: {single_file_path}")
        state_dict = load_file(single_file_path)
    else:
        raise FileNotFoundError(
            f"Could not find model weights at {checkpoint_path}. "
            f"Expected either {index_path} or {single_file_path}"
        )
    
    # Load the state dict into the model
    missing_keys, unexpected_keys = multimodal_model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {missing_keys[:10]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys[:10]}...")
    
    logger.info("Octopus model loaded successfully from checkpoint")
    
    # NOW apply LoRA wrapper if needed (after loading weights)
    if model_args.use_peft:
        from peft import get_peft_model
        logger.info("Applying LoRA wrapper to LLM")
        peft_config = get_peft_config(model_args)
        if peft_config is not None:
            # Wrap the inner LLM with LoRA
            multimodal_model.llm_model = get_peft_model(multimodal_model.llm_model, peft_config)
    
    # Apply freezing settings
    freeze_config = {
        'freeze_encoder': getattr(multimodal_config, 'freeze_encoder', False),
        'freeze_llm': getattr(multimodal_config, 'freeze_llm', False),
        'freeze_gates': getattr(multimodal_config, 'freeze_gates', False),
        'freeze_fusion_blocks': getattr(multimodal_config, 'freeze_fusion_blocks', False),
        'freeze_projections': getattr(multimodal_config, 'freeze_projections', False),
    }
    multimodal_model.apply_freezing_config(freeze_config)
    
    return multimodal_model


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
    
    Checkpoint Loading:
    - prepared_checkpoint_path: Load from a prepared checkpoint using config-first approach
    - octopus_checkpoint_path: Load from a full Octopus checkpoint with from_pretrained
    - If neither is set, creates a new model from scratch
    
    Args:
        model_args: Model configuration from TRL
        training_args: Training configuration
        multimodal_config: Multi-modal specific configuration (includes freezing options
                          and optional checkpoint paths)
    
    Returns:
        Octopus instance with the specified configuration and freezing applied
    """
    # Check if we should load from a prepared checkpoint (config-first loading)
    prepared_checkpoint = getattr(multimodal_config, "prepared_checkpoint_path", None)
    
    if prepared_checkpoint is not None:
        logger.info(f"Loading prepared Octopus model from checkpoint: {prepared_checkpoint}")
        return load_prepared_octopus_from_checkpoint(
            prepared_checkpoint,
            model_args,
            multimodal_config,
        )
    
    # Check if we should load from a full Octopus checkpoint
    octopus_checkpoint = getattr(multimodal_config, "octopus_checkpoint_path", None)
    
    if octopus_checkpoint is not None:
        logger.info(f"Loading trained Octopus model from checkpoint: {octopus_checkpoint}")
        return _load_octopus_from_checkpoint(
            octopus_checkpoint, 
            model_args, 
            training_args, 
            multimodal_config
        )
    
    # Otherwise, create new Octopus model from scratch
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
    
    # Determine encoder hidden dimension
    # If encoder_hidden_dim is explicitly set, use it (for loading pretrained encoders with different dims)
    # Otherwise, use modality_embedding_dim for both encoder and fusion
    encoder_dim = (
        multimodal_config.encoder_hidden_dim
        if multimodal_config.encoder_hidden_dim is not None
        else multimodal_config.modality_embedding_dim
    )
    
    # Determine fusion hidden dimension
    fusion_dim = (
        multimodal_config.fusion_hidden_dim
        if multimodal_config.fusion_hidden_dim is not None
        else multimodal_config.modality_embedding_dim
    )
    
    mm_config = OctopusConfig(
        encoder=EncoderConfig(hidden_dim=int(encoder_dim)),
        patching=PatchingConfig(
            k_max=int(multimodal_config.k_max),
            r_max=multimodal_config.r_max,
            dynamic_k_mass=multimodal_config.dynamic_k_mass,
            beta=float(multimodal_config.beta),
            tau=float(multimodal_config.tau),
        ),
        fusion=FusionConfig(
            num_blocks=int(multimodal_config.num_fusion_blocks),
            num_heads=int(multimodal_config.num_attention_heads),
            hidden_dim=int(fusion_dim),
            intermediate_dim=multimodal_config.fusion_intermediate_dim,
            dropout=float(multimodal_config.dropout),
        ),
        prediction_head=PredictionHeadConfig(
            task_type=getattr(multimodal_config, 'task_type', None),
            num_labels=getattr(multimodal_config, 'num_labels', 2),
            pooling_strategy=getattr(multimodal_config, 'pooling_strategy', 'last'),
            hidden_dim=getattr(multimodal_config, 'head_hidden_dim', None),
            dropout=getattr(multimodal_config, 'head_dropout', 0.1),
            use_dual_loss=getattr(multimodal_config, 'use_dual_loss', False),
            lm_loss_weight=getattr(multimodal_config, 'lm_loss_weight', 0.5),
        ),
    )

    # Log prediction head config if enabled
    if mm_config.prediction_head.task_type is not None:
        logger.info(f"Configuring prediction head: task_type={mm_config.prediction_head.task_type}, "
                   f"use_dual_loss={mm_config.prediction_head.use_dual_loss}, "
                   f"lm_loss_weight={mm_config.prediction_head.lm_loss_weight}")

    # Create the multi-modal wrapper with the (possibly PEFT-wrapped) LLM
    multimodal_model = Octopus(llm_model=llm_model, config=mm_config)
    
    # Load pretrained encoder weights if checkpoint path is provided
    if hasattr(multimodal_config, 'encoder_checkpoint_path') and multimodal_config.encoder_checkpoint_path is not None:
        encoder_checkpoint = multimodal_config.encoder_checkpoint_path
        print(f"\n{'='*80}")
        print(f"LOADING PRETRAINED ENCODER FROM CHECKPOINT")
        print(f"{'='*80}")
        print(f"Checkpoint path: {encoder_checkpoint}")
        logger.info(f"Loading pretrained encoder from checkpoint: {encoder_checkpoint}")
        try:
            load_encoder_weights_from_checkpoint(multimodal_model.encoder, encoder_checkpoint)
            print(f"✓ Successfully loaded encoder weights from {encoder_checkpoint}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"✗ Failed to load encoder checkpoint: {e}")
            logger.error(f"Failed to load encoder checkpoint: {e}")
            raise
    
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
