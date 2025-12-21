"""
Training Script for Masked Reconstruction Pre-training

Uses HuggingFace Transformers Trainer for graph encoder pre-training.
Supports config file loading from YAML and wandb reporting.
"""

import os
import torch
import logging
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from transformers import TrainingArguments, Trainer, HfArgumentParser

from src.models.geo_encoder import GeoEncoder
from src.trainer.masked_reconstruction import MaskedReconstructionTrainer
from src.data_loader.graph_dataset import GraphDataset, GraphBatchCollator

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("wandb not available. Install with: pip install wandb")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    hidden_dim: int = field(default=256, metadata={"help": "Hidden dimension for encoder"})
    num_layers: int = field(default=6, metadata={"help": "Number of EGNN layers"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate"})
    update_coords: bool = field(default=True, metadata={"help": "Update coordinates in EGNN"})
    use_layernorm: bool = field(default=True, metadata={"help": "Use layer normalization"})
    num_rbf: int = field(default=32, metadata={"help": "Number of RBF kernels"})
    rbf_max: float = field(default=10.0, metadata={"help": "Max distance for RBF"})
    
    # Reconstruction head parameters
    num_elements: int = field(default=119, metadata={"help": "Number of element types"})
    num_dist_bins: int = field(default=64, metadata={"help": "Number of distance bins"})
    dist_min: float = field(default=0.0, metadata={"help": "Min distance for binning"})
    dist_max: float = field(default=20.0, metadata={"help": "Max distance for binning"})
    
    # Loss weights
    element_weight: float = field(default=1.0, metadata={"help": "Element loss weight"})
    dist_weight: float = field(default=1.0, metadata={"help": "Distance loss weight"})
    noise_weight: float = field(default=1.0, metadata={"help": "Noise loss weight"})


@dataclass
class DataArguments:
    """Arguments for data loading."""
    
    train_data_path: Optional[str] = field(default=None, metadata={"help": "Path to training parquet file"})
    val_data_path: Optional[str] = field(default=None, metadata={"help": "Path to validation parquet file"})
    val_split_ratio: float = field(default=0.1, metadata={"help": "Validation split ratio when auto-splitting"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for datasets"})
    
    # Masking parameters
    node_mask_prob: float = field(default=0.15, metadata={"help": "Node masking probability"})
    noise_std: float = field(default=0.1, metadata={"help": "Coordinate noise std dev"})
    use_soft_dist_targets: bool = field(default=False, metadata={"help": "Use soft distance targets"})
    soft_dist_sigma: float = field(default=0.5, metadata={"help": "Sigma for soft distance targets"})


@dataclass
class ScriptArguments:
    """Arguments for script configuration."""
    
    config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML config file. If provided, will override other arguments."}
    )
    wandb_project: Optional[str] = field(
        default="EntroAdap-MaskedReconstruction",
        metadata={"help": "Wandb project name"}
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb entity/team name"}
    )
    wandb_tags: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated wandb tags"}
    )
    wandb_notes: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb run notes"}
    )


class MaskedReconstructionTrainerWrapper(Trainer):
    """
    Custom Trainer wrapper for masked reconstruction.
    
    Handles the specific forward pass and loss computation for graph reconstruction.
    """
    
    def _move_to_device(self, data, device):
        """
        Recursively move nested data structures to device.
        
        Args:
            data: Data structure (dict, list, tensor, or primitive)
            device: Target device
        
        Returns:
            Data structure with all tensors moved to device
        """
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: self._move_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._move_to_device(item, device) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._move_to_device(item, device) for item in data)
        else:
            return data
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for masked reconstruction.
        
        Args:
            model: MaskedReconstructionTrainer instance
            inputs: Batch dictionary from GraphBatchCollator
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        
        Returns:
            loss (and optionally outputs)
        """
        # Get device from model
        device = next(model.parameters()).device
        
        # Move ALL inputs to device (including nested structures)
        data = self._move_to_device(inputs['data'], device)
        batch = inputs['batch'].to(device)
        node_mask = inputs['node_mask'].to(device)
        edge_mask = inputs['edge_mask'].to(device)
        element_labels = inputs['element_labels'].to(device)
        dist_labels = inputs['dist_labels'].to(device)
        noise_labels = inputs['noise_labels'].to(device)
        
        # Forward pass
        outputs = model(
            data=data,
            batch=batch,
            node_mask=node_mask,
            edge_mask=edge_mask,
            element_labels=element_labels,
            dist_labels=dist_labels,
            noise_labels=noise_labels,
            compute_loss=True,
        )
        
        loss = outputs['loss']
        
        # Log individual losses
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            log_dict = {}
            if 'element_loss' in outputs:
                log_dict['element_loss'] = outputs['element_loss'].item()
            if 'dist_loss' in outputs:
                log_dict['dist_loss'] = outputs['dist_loss'].item()
            if 'noise_loss' in outputs:
                log_dict['noise_loss'] = outputs['noise_loss'].item()
            if log_dict:
                self.log(log_dict)
        
        return (loss, outputs) if return_outputs else loss


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(
    config: Dict[str, Any],
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> tuple:
    """
    Merge YAML config with command-line arguments.
    Config file takes precedence over defaults, but CLI args take precedence over config.
    
    Args:
        config: Configuration dictionary from YAML
        model_args: Model arguments (potentially with CLI overrides)
        data_args: Data arguments (potentially with CLI overrides)
        training_args: Training arguments (potentially with CLI overrides)
    
    Returns:
        Tuple of (model_args, data_args, training_args) with merged values
    """
    # Update model args from config
    if 'model' in config:
        for key, value in config['model'].items():
            if hasattr(model_args, key):
                setattr(model_args, key, value)
    
    # Update data args from config
    if 'data' in config:
        for key, value in config['data'].items():
            if hasattr(data_args, key):
                setattr(data_args, key, value)
    
    # Update training args from config
    if 'training' in config:
        training_dict = vars(training_args)
        for key, value in config['training'].items():
            if key in training_dict:
                setattr(training_args, key, value)
    
    return model_args, data_args, training_args


def setup_wandb(config: Dict[str, Any], script_args: ScriptArguments, training_args: TrainingArguments):
    """
    Setup wandb configuration.
    
    Args:
        config: Configuration dictionary
        script_args: Script arguments
        training_args: Training arguments
    """
    if not WANDB_AVAILABLE:
        logger.warning("Wandb is not available. Skipping wandb setup.")
        return
    
    # Get wandb config from YAML or script args
    wandb_config = config.get('wandb', {})
    
    project = script_args.wandb_project or wandb_config.get('project', 'EntroAdap-MaskedReconstruction')
    entity = script_args.wandb_entity or wandb_config.get('entity', None)
    tags = script_args.wandb_tags.split(',') if script_args.wandb_tags else wandb_config.get('tags', [])
    notes = script_args.wandb_notes or wandb_config.get('notes', None)
    
    # Set environment variables for wandb
    if project:
        os.environ['WANDB_PROJECT'] = project
    if entity:
        os.environ['WANDB_ENTITY'] = entity
    if tags:
        os.environ['WANDB_TAGS'] = ','.join(tags)
    if notes:
        os.environ['WANDB_NOTES'] = notes
    
    # Update training args to use wandb
    if 'wandb' not in training_args.report_to:
        training_args.report_to = ['wandb']
    
    logger.info(f"Wandb reporting enabled: project={project}, entity={entity}, tags={tags}")


def main():
    """Main training function."""
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ScriptArguments))
    model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses()
    
    # Load config file if provided
    config = {}
    if script_args.config:
        logger.info(f"Loading configuration from {script_args.config}")
        config = load_config_from_yaml(script_args.config)
        model_args, data_args, training_args = merge_config_with_args(
            config, model_args, data_args, training_args
        )
    
    # Validate required arguments
    if data_args.train_data_path is None:
        raise ValueError(
            "train_data_path is required. Please provide it via --train_data_path CLI argument "
            "or in the config file under data.train_data_path"
        )
    
    # Setup wandb if enabled
    if 'wandb' in training_args.report_to or (config.get('training', {}).get('report_to') and 'wandb' in config['training']['report_to']):
        setup_wandb(config, script_args, training_args)
    
    # Auto-generate run name if not provided
    if training_args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_args.run_name = f"masked_reconstruction_{timestamp}"
    
    # Set up logging
    logger.info(f"Training arguments: {training_args}")
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    
    # Create model
    logger.info("Creating model...")
    encoder = GeoEncoder(
        hidden_dim=model_args.hidden_dim,
        num_layers=model_args.num_layers,
        dropout=model_args.dropout,
        update_coords=model_args.update_coords,
        use_layernorm=model_args.use_layernorm,
        num_rbf=model_args.num_rbf,
        rbf_max=model_args.rbf_max,
    )
    
    model = MaskedReconstructionTrainer(
        encoder=encoder,
        num_elements=model_args.num_elements,
        num_dist_bins=model_args.num_dist_bins,
        dist_min=model_args.dist_min,
        dist_max=model_args.dist_max,
        element_weight=model_args.element_weight,
        dist_weight=model_args.dist_weight,
        noise_weight=model_args.noise_weight,
    )
    
    # Create datasets
    logger.info("Loading datasets...")
    
    if data_args.val_data_path:
        # Load separate train and validation datasets
        logger.info(f"Loading training data from {data_args.train_data_path}")
        train_dataset = GraphDataset(
            dataset_path=data_args.train_data_path,
            split='train',
            cache_dir=data_args.cache_dir,
        )
        
        logger.info(f"Loading validation data from {data_args.val_data_path}")
        eval_dataset = GraphDataset(
            dataset_path=data_args.val_data_path,
            split='validation',
            cache_dir=data_args.cache_dir,
        )
    else:
        # Auto-split validation set from training data
        logger.info(f"Loading data from {data_args.train_data_path}")
        logger.info(f"Auto-splitting validation set with ratio {data_args.val_split_ratio}")
        
        # Load full dataset
        full_dataset = GraphDataset(
            dataset_path=data_args.train_data_path,
            split='train',
            cache_dir=data_args.cache_dir,
        )
        
        # Split into train and validation
        total_size = len(full_dataset)
        val_size = int(total_size * data_args.val_split_ratio)
        train_size = total_size - val_size
        
        # Use torch.utils.data.random_split for deterministic splitting with seed
        generator = torch.Generator().manual_seed(training_args.seed)
        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=generator,
        )
        
        logger.info(f"Split dataset: {train_size} train samples, {val_size} validation samples")
    
    # Create collator
    collator = GraphBatchCollator(
        node_mask_prob=data_args.node_mask_prob,
        noise_std=data_args.noise_std,
        num_dist_bins=model_args.num_dist_bins,
        dist_min=model_args.dist_min,
        dist_max=model_args.dist_max,
        use_soft_dist_targets=data_args.use_soft_dist_targets,
        soft_dist_sigma=data_args.soft_dist_sigma,
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = MaskedReconstructionTrainerWrapper(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {training_args.output_dir}...")
    trainer.save_model()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

