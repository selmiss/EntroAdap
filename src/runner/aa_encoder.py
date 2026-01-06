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
from transformers import TrainingArguments, Trainer, HfArgumentParser, TrainerCallback

from src.models.aa_encoder import AAEncoder
from src.trainer.reconstruction import ReconstructionTrainer
from src.data_loader.aa_dataset import GraphDataset, GraphBatchCollator, ModalityAwareBatchSampler

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
    
    train_data_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path(s) to training parquet file(s). Supports: single path, comma-separated paths, or list in YAML config"}
    )
    val_data_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path(s) to validation parquet file(s). Supports: single path, comma-separated paths, or list in YAML config"}
    )
    val_split_ratio: float = field(default=0.1, metadata={"help": "Validation split ratio when auto-splitting"})
    stratified_val_split: bool = field(
        default=True,
        metadata={"help": "Use stratified validation split (each dataset contributes proportionally to val set)"}
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for datasets"})
    use_modality_sampler: bool = field(
        default=True, 
        metadata={"help": "Use modality-aware batch sampler to ensure same-modality batches (required for multi-modality datasets)"}
    )
    max_samples_per_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Max samples per dataset. Can be: single int (applies to all), comma-separated ints, or list in YAML"}
    )
    
    # Runtime filtering thresholds
    max_atoms: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum atoms per structure. Structures exceeding this will be skipped at runtime. None means no limit."}
    )
    max_edges: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum edges per structure. Structures exceeding this will be skipped at runtime. None means no limit."}
    )
    skip_on_error: bool = field(
        default=True,
        metadata={"help": "Skip samples that fail to load or exceed thresholds (instead of raising errors)"}
    )
    
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
        default=None,
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


class LossLoggingCallback(TrainerCallback):
    """
    Callback to log individual loss components during training and evaluation.
    
    This properly handles gradient accumulation by only logging after the optimizer step.
    """
    
    def __init__(self):
        super().__init__()
        self.accumulated_train_losses = {'element': [], 'dist': [], 'noise': []}
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step (after gradient accumulation and optimizer step)."""
        # Log accumulated losses if we have any
        if any(self.accumulated_train_losses.values()):
            log_dict = {}
            if self.accumulated_train_losses['element']:
                log_dict['train_element_loss'] = sum(self.accumulated_train_losses['element']) / len(self.accumulated_train_losses['element'])
            if self.accumulated_train_losses['dist']:
                log_dict['train_dist_loss'] = sum(self.accumulated_train_losses['dist']) / len(self.accumulated_train_losses['dist'])
            if self.accumulated_train_losses['noise']:
                log_dict['train_noise_loss'] = sum(self.accumulated_train_losses['noise']) / len(self.accumulated_train_losses['noise'])
            
            # Log to wandb/tensorboard via the trainer
            if log_dict and kwargs.get('logs') is not None:
                kwargs['logs'].update(log_dict)
            
            # Clear accumulated losses
            self.accumulated_train_losses = {'element': [], 'dist': [], 'noise': []}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging (respects logging_steps)."""
        # This is where we can add custom metrics to logs
        pass
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics:
            # Log evaluation losses with proper prefixes
            logger.info(f"Evaluation at step {state.global_step}:")
            for key, value in metrics.items():
                if 'eval_loss' in key or 'eval' in key:
                    logger.info(f"  {key}: {value}")


class ReconstructionTrainerWrapper(Trainer):
    """
    Custom Trainer wrapper for masked reconstruction.
    
    Handles the specific forward pass and loss computation for graph reconstruction.
    Supports modality-aware batch sampling for mixed-modality training.
    """
    
    def __init__(self, *args, use_modality_sampler=False, **kwargs):
        """
        Args:
            use_modality_sampler: Whether to use ModalityAwareBatchSampler
        """
        self.use_modality_sampler = use_modality_sampler
        # Track individual losses for logging (accumulated across gradient accumulation steps)
        self.accumulated_losses = {'element': [], 'dist': [], 'noise': []}
        super().__init__(*args, **kwargs)
    
    def get_train_dataloader(self):
        """
        Returns the training DataLoader with optional modality-aware batch sampler.
        """
        if self.use_modality_sampler:
            from torch.utils.data import DataLoader
            
            # Create modality-aware batch sampler
            batch_sampler = ModalityAwareBatchSampler(
                dataset=self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )
            
            # Create DataLoader with batch_sampler
            # Note: when using batch_sampler, we can't specify batch_size, shuffle, or drop_last
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            # Use default behavior
            return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Returns the evaluation DataLoader with optional modality-aware batch sampler.
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if self.use_modality_sampler and eval_dataset is not None:
            from torch.utils.data import DataLoader
            
            # Create modality-aware batch sampler (no shuffle for eval)
            batch_sampler = ModalityAwareBatchSampler(
                dataset=eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )
            
            return DataLoader(
                eval_dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            # Use default behavior
            return super().get_eval_dataloader(eval_dataset)
    
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
            model: ReconstructionTrainer instance
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
        
        # Accumulate individual losses during training (for logging after gradient accumulation)
        if model.training:
            if 'element_loss' in outputs:
                self.accumulated_losses['element'].append(outputs['element_loss'].item())
            if 'dist_loss' in outputs:
                self.accumulated_losses['dist'].append(outputs['dist_loss'].item())
            if 'noise_loss' in outputs:
                self.accumulated_losses['noise'].append(outputs['noise_loss'].item())
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to add individual loss logging.
        """
        # Get eval dataloader
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            return {}
        
        # Run parent evaluation
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        
        # Manually compute individual losses
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()
        
        element_losses = []
        dist_losses = []
        noise_losses = []
        
        with torch.no_grad():
            for inputs in eval_dataloader:
                # Compute loss with outputs
                _, outputs = self.compute_loss(model, inputs, return_outputs=True)
                
                if 'element_loss' in outputs:
                    element_losses.append(outputs['element_loss'].item())
                if 'dist_loss' in outputs:
                    dist_losses.append(outputs['dist_loss'].item())
                if 'noise_loss' in outputs:
                    noise_losses.append(outputs['noise_loss'].item())
        
        # Add individual loss metrics
        if element_losses:
            metrics[f'{metric_key_prefix}_element_loss'] = sum(element_losses) / len(element_losses)
            metrics[f'{metric_key_prefix}_dist_loss'] = sum(dist_losses) / len(dist_losses)
            metrics[f'{metric_key_prefix}_noise_loss'] = sum(noise_losses) / len(noise_losses)
        
        return metrics
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log method to add accumulated individual losses.
        This is called by Trainer after gradient accumulation is complete.
        """
        # Add accumulated losses to logs if we have any (check if in training by looking at accumulated losses)
        if self.accumulated_losses['element']:
            logs['train_element_loss'] = sum(self.accumulated_losses['element']) / len(self.accumulated_losses['element'])
            logs['train_dist_loss'] = sum(self.accumulated_losses['dist']) / len(self.accumulated_losses['dist'])
            logs['train_noise_loss'] = sum(self.accumulated_losses['noise']) / len(self.accumulated_losses['noise'])
            
            # Clear accumulated losses after logging
            self.accumulated_losses = {'element': [], 'dist': [], 'noise': []}
        
        # Call parent log method with exact signature
        super().log(logs, start_time)


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
    
    project = script_args.wandb_project or wandb_config.get('project', 'EntroAdap-Reconstruction')
    entity = script_args.wandb_entity or wandb_config.get('entity', None)
    mode = wandb_config.get('mode', 'online')  # Default to online if not specified
    tags = script_args.wandb_tags.split(',') if script_args.wandb_tags else wandb_config.get('tags', [])
    notes = script_args.wandb_notes or wandb_config.get('notes', None)
    
    # Set environment variables for wandb
    if project:
        os.environ['WANDB_PROJECT'] = project
    if entity:
        os.environ['WANDB_ENTITY'] = entity
    if mode:
        os.environ['WANDB_MODE'] = mode
    if tags:
        os.environ['WANDB_TAGS'] = ','.join(tags)
    if notes:
        os.environ['WANDB_NOTES'] = notes
    
    # Update training args to use wandb
    if 'wandb' not in training_args.report_to:
        training_args.report_to = ['wandb']
    
    logger.info(f"Wandb reporting enabled: project={project}, entity={entity}, mode={mode}, tags={tags}")


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
    encoder = AAEncoder(
        hidden_dim=model_args.hidden_dim,
        num_layers=model_args.num_layers,
        dropout=model_args.dropout,
        update_coords=model_args.update_coords,
        use_layernorm=model_args.use_layernorm,
        num_rbf=model_args.num_rbf,
        rbf_max=model_args.rbf_max,
    )
    
    model = ReconstructionTrainer(
        encoder=encoder,
        num_elements=model_args.num_elements,
        num_dist_bins=model_args.num_dist_bins,
        dist_min=model_args.dist_min,
        dist_max=model_args.dist_max,
        element_weight=model_args.element_weight,
        dist_weight=model_args.dist_weight,
        noise_weight=model_args.noise_weight,
    )
    
    # Parse max_samples_per_dataset
    max_samples_list = None
    if data_args.max_samples_per_dataset is not None:
        if isinstance(data_args.max_samples_per_dataset, str):
            # Parse comma-separated string
            max_samples_list = [
                int(x.strip()) if x.strip().lower() != 'none' else None 
                for x in data_args.max_samples_per_dataset.split(',')
            ]
        elif isinstance(data_args.max_samples_per_dataset, (int, list)):
            max_samples_list = data_args.max_samples_per_dataset
    
    # Create datasets
    logger.info("Loading datasets...")
    
    if data_args.val_data_path:
        # Load separate train and validation datasets
        logger.info(f"Loading training data from {data_args.train_data_path}")
        train_dataset = GraphDataset(
            dataset_path=data_args.train_data_path,
            split='train',
            cache_dir=data_args.cache_dir,
            max_samples_per_dataset=max_samples_list,
            max_atoms=data_args.max_atoms,
            max_edges=data_args.max_edges,
            skip_on_error=data_args.skip_on_error,
        )
        
        logger.info(f"Loading validation data from {data_args.val_data_path}")
        eval_dataset = GraphDataset(
            dataset_path=data_args.val_data_path,
            split='validation',
            cache_dir=data_args.cache_dir,
            max_samples_per_dataset=max_samples_list,
            max_atoms=data_args.max_atoms,
            max_edges=data_args.max_edges,
            skip_on_error=data_args.skip_on_error,
        )
    else:
        # Auto-split validation set from training data
        logger.info(f"Auto-splitting validation set with ratio {data_args.val_split_ratio}")
        
        # Check if using stratified split for multi-dataset training
        is_multi_dataset = isinstance(data_args.train_data_path, list) or (
            isinstance(data_args.train_data_path, str) and ',' in data_args.train_data_path
        )
        
        if is_multi_dataset and data_args.stratified_val_split:
            logger.info("Using stratified validation split (proportional sampling from each dataset)")
            
            # Load full concatenated dataset with max_samples applied
            # This uses cache and only loads once
            full_dataset = GraphDataset(
                dataset_path=data_args.train_data_path,
                split='train',
                cache_dir=data_args.cache_dir,
                max_samples_per_dataset=max_samples_list,
                max_atoms=data_args.max_atoms,
                max_edges=data_args.max_edges,
                skip_on_error=data_args.skip_on_error,
            )
            
            # Get the underlying HF dataset
            hf_dataset = full_dataset.dataset
            
            # Group indices by modality for stratified split
            print("\n" + "="*80)
            print("Creating Stratified Train/Val Split:")
            print("="*80)
            
            modality_indices = {}
            print("Grouping samples by modality...")
            
            # OPTIMIZED: Bulk read all modalities at once (much faster than per-sample access)
            all_modalities = hf_dataset['modality']
            for idx, modality in enumerate(all_modalities):
                if modality not in modality_indices:
                    modality_indices[modality] = []
                modality_indices[modality].append(idx)
            
            # Split each modality proportionally
            train_indices = []
            eval_indices = []
            
            for modality, indices in modality_indices.items():
                total_size = len(indices)
                val_size = int(total_size * data_args.val_split_ratio)
                train_size = total_size - val_size
                
                # Shuffle with seed for reproducibility
                generator = torch.Generator().manual_seed(training_args.seed)
                shuffled_indices = torch.randperm(total_size, generator=generator).tolist()
                
                # Split indices
                train_modal_indices = [indices[i] for i in shuffled_indices[:train_size]]
                eval_modal_indices = [indices[i] for i in shuffled_indices[train_size:]]
                
                train_indices.extend(train_modal_indices)
                eval_indices.extend(eval_modal_indices)
                
                print(f"  {modality}: {total_size:,} total → Train: {train_size:,} | Val: {val_size:,}")
            
            # Create subset datasets using indices
            from torch.utils.data import Subset
            train_dataset = Subset(full_dataset, train_indices)
            eval_dataset = Subset(full_dataset, eval_indices)
            
            print("-" * 80)
            print(f"Total Train: {len(train_dataset):,} samples | Total Val: {len(eval_dataset):,} samples")
            print("="*80 + "\n")
            
        else:
            # Original non-stratified split
            logger.info("Using simple random split (non-stratified)")
            
            # Load full dataset
            full_dataset = GraphDataset(
                dataset_path=data_args.train_data_path,
                split='train',
                cache_dir=data_args.cache_dir,
                max_samples_per_dataset=max_samples_list,
                max_atoms=data_args.max_atoms,
                max_edges=data_args.max_edges,
                skip_on_error=data_args.skip_on_error,
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
    
    # Create loss logging callback
    loss_callback = LossLoggingCallback()
    
    trainer = ReconstructionTrainerWrapper(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        use_modality_sampler=data_args.use_modality_sampler,
        callbacks=[loss_callback],
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save evaluation results
        import json
        eval_output_path = os.path.join(training_args.output_dir, "eval_results.json")
        with open(eval_output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Evaluation results saved to {eval_output_path}")
    
    # Save final model
    logger.info(f"Saving model to {training_args.output_dir}...")
    trainer.save_model()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

