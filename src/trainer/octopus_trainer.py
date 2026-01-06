"""
Custom SFT Trainer for Octopus that properly passes graph data.
"""

from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from trl import SFTTrainer


class MultiModalSFTTrainer(SFTTrainer):
    """
    Custom SFT Trainer that overrides compute_loss to pass graph_data
    and related arguments to Octopus.forward().
    
    Also provides custom dataloader with ModalityAwareBatchSampler to ensure
    batches don't mix different modalities (e.g., DNA and RNA).
    """
    def _prepare_dataset(self, dataset, *args, **kwargs):
        """
        Override TRL's dataset preparation to prevent truncation of graph features.
        """

        return dataset

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Override log to normalize loss for comparison with eval_loss.
        
        The default 'loss' logged by HuggingFace Trainer is the accumulated loss
        over gradient_accumulation_steps, making it incomparable with eval_loss.
        We normalize it by dividing by gradient_accumulation_steps.
        """
        if 'loss' in logs and self.args.gradient_accumulation_steps > 1:
            # Keep original accumulated loss for reference
            logs['loss_accumulated'] = logs['loss']
            # Replace loss with normalized value that's comparable to eval_loss
            logs['loss'] = logs['loss'] / self.args.gradient_accumulation_steps
        
        super().log(logs, start_time=start_time)
    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader with a custom batch sampler that groups
        samples by modality to prevent mixing modalities in a single batch.
        """
        from src.data_loader import ModalityAwareBatchSamplerForSFT
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        # Check if dataset has modality information
        has_modality_info = (
            hasattr(train_dataset, 'column_names') and 
            ('modality' in train_dataset.column_names or 'graph_data' in train_dataset.column_names)
        )
        
        if has_modality_info:
            # Use custom batch sampler to group by modality
            batch_sampler = ModalityAwareBatchSamplerForSFT(
                dataset=train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last,
            )
            
            # Note: persistent_workers is incompatible with batch_sampler
            return DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                # persistent_workers can't be used with batch_sampler
            )
        else:
            # Fall back to default behavior for datasets without modality
            return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """
        Returns the evaluation DataLoader with a custom batch sampler that groups
        samples by modality to prevent mixing modalities in a single batch.
        """
        from src.data_loader import ModalityAwareBatchSamplerForSFT
        
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        data_collator = self.data_collator
        
        # Check if dataset has modality information
        has_modality_info = (
            hasattr(eval_dataset, 'column_names') and 
            ('modality' in eval_dataset.column_names or 'graph_data' in eval_dataset.column_names)
        )
        
        if has_modality_info:
            # Use custom batch sampler to group by modality
            batch_sampler = ModalityAwareBatchSamplerForSFT(
                dataset=eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,  # Don't shuffle eval data
                seed=self.args.seed,
                drop_last=False,  # Keep all eval samples
            )
            
            # Note: persistent_workers is incompatible with batch_sampler
            return DataLoader(
                eval_dataset,
                batch_sampler=batch_sampler,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                # persistent_workers can't be used with batch_sampler
            )
        else:
            # Fall back to default behavior for datasets without modality
            return super().get_eval_dataloader(eval_dataset)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for multimodal training, passing graph data to model.
        
        Args:
            model: Octopus instance
            inputs: Batch dictionary from MultiModalDataCollator with:
                - input_ids, attention_mask, labels (standard)
                - graph_data, batch, instr_positions, patch_positions (multimodal)
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch
        
        Returns:
            loss (and optionally outputs)
        """
        # Extract multimodal-specific inputs
        graph_data = inputs.pop("graph_data", None)
        batch = inputs.pop("batch", None)
        instr_positions = inputs.pop("instr_positions", None)
        patch_positions = inputs.pop("patch_positions", None)
        patch_mask = inputs.pop("patch_mask", None)
        node_mask = inputs.pop("node_mask", None)
        
        # Remove internal tracking fields
        inputs.pop("_graph_indices", None)
        
        # Call model with all arguments
        outputs = model(
            **inputs,
            graph_data=graph_data,
            batch=batch,
            instr_positions=instr_positions,
            patch_positions=patch_positions,
            patch_mask=patch_mask,
            node_mask=node_mask,
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

