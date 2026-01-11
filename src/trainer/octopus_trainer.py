"""
Custom SFT Trainer for Octopus that properly passes graph data.
"""

from typing import Dict, Any, Optional, Union
import torch
from torch.utils.data import DataLoader
from trl import SFTTrainer
import numpy as np
from src.trainer.metrics.metrics_diverting import compute_metrics


class MultiModalSFTTrainer(SFTTrainer):
    """
    Custom SFT Trainer that overrides compute_loss to pass graph_data
    and related arguments to Octopus.forward().
    
    Also provides custom dataloader with ModalityAwareBatchSampler to ensure
    batches don't mix different modalities (e.g., DNA and RNA).
    """
    
    def __init__(self, *args, eval_metrics: str = "none", **kwargs):
        """
        Initialize trainer with optional metrics computation.
        
        Args:
            eval_metrics: Evaluation metrics to compute - 'text', 'qa', 'molgen', 'mae', or 'none'
        """
        super().__init__(*args, **kwargs)
        self.eval_metrics = eval_metrics
        self._eval_predictions = []
        self._eval_labels = []
        self._eval_prompts = []
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
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to collect predictions and labels for metrics.
        Uses actual generation instead of argmax for better quality predictions.
        """
        # If any metrics are enabled, we need to generate text
        need_generation_for_metrics = self.eval_metrics != "none" and not model.training
        
        # Ensure inputs are on the correct device using trainer's method
        inputs = self._prepare_inputs(inputs)
        
        # Make a copy of inputs before compute_loss mutates it (if we need it for generation)
        inputs_for_generation = inputs.copy() if need_generation_for_metrics else None
        
        # Always compute loss for evaluation (prediction_loss_only means we only need loss, not logits)
        with torch.no_grad():
            outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
        
        # Generate predictions if metrics are enabled
        if need_generation_for_metrics:
            with torch.no_grad():
                # Truncate input_ids to prompt only (match inference behavior)
                # In training, input_ids includes full sequence (prompt + assistant response)
                # But for generation, we should only use prompt (where labels == -100)
                labels = inputs_for_generation["labels"]
                
                # Find where assistant response starts (first non -100 label per sample)
                # This keeps the assistant header but removes the response
                batch_size = labels.shape[0]
                truncated_inputs = {}
                
                for key in inputs_for_generation.keys():
                    if key in ["input_ids", "attention_mask"]:
                        # Truncate to longest prompt in batch
                        prompt_lengths = (labels == -100).sum(dim=1)
                        max_prompt_len = prompt_lengths.max().item()
                        truncated_inputs[key] = inputs_for_generation[key][:, :max_prompt_len]
                    elif key == "labels":
                        # Don't include labels in generation
                        continue
                    elif key == "patch_positions":
                        # patch_positions are relative to full sequence, no adjustment needed
                        # They point to structure token position which is in the prompt
                        truncated_inputs[key] = inputs_for_generation[key]
                    else:
                        # Pass through other fields (graph_data, batch, etc.)
                        truncated_inputs[key] = inputs_for_generation[key]
                
                # Generate with prompt-only inputs (matching inference behavior)
                generated_ids = model.generate(**truncated_inputs)
                
                # Convert to numpy
                pred_ids_np = generated_ids.cpu().numpy()
                labels_np = inputs_for_generation["labels"].cpu().numpy() if isinstance(inputs_for_generation["labels"], torch.Tensor) else inputs_for_generation["labels"]
                input_ids_np = inputs_for_generation["input_ids"].cpu().numpy() if isinstance(inputs_for_generation["input_ids"], torch.Tensor) else inputs_for_generation["input_ids"]
                
                # Store each sample individually
                for pred, label, input_ids in zip(pred_ids_np, labels_np, input_ids_np):
                    # Extract prompt only (where labels are -100, indicating ignored tokens)
                    # The prompt is the part before the answer starts
                    prompt_mask = (label == -100)
                    prompt_ids = input_ids[prompt_mask] if prompt_mask.any() else input_ids
                    
                    self._eval_predictions.append(pred)
                    self._eval_labels.append(label)
                    self._eval_prompts.append(prompt_ids)
        
        # Return according to prediction_loss_only flag
        if prediction_loss_only:
            return (loss, None, None)
        
        # For compatibility, return None for logits when using generation
        return (loss, None, inputs.get("labels"))
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to compute and log text metrics after evaluation.
        """
        # Reset stored predictions and labels
        self._eval_predictions = []
        self._eval_labels = []
        self._eval_prompts = []
        
        # Get the dataset that will be evaluated
        dataset_to_eval = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Run standard evaluation
        output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Compute metrics based on eval_metrics setting
        if self.eval_metrics != "none" and self._eval_predictions:
            try:
                # Extract categories from dataset if available
                categories = None
                if dataset_to_eval is not None and hasattr(dataset_to_eval, 'column_names'):
                    if 'category' in dataset_to_eval.column_names:
                        categories = dataset_to_eval['category']
                    elif 'task' in dataset_to_eval.column_names:
                        categories = dataset_to_eval['task']
                
                metrics, detailed_results = compute_metrics(
                    self.eval_metrics,
                    self._eval_predictions,
                    self._eval_labels,
                    self.processing_class,
                    metric_key_prefix,
                    categories,
                    prompts=self._eval_prompts
                )
                
                # Add metrics to output with appropriate prefix
                for key, value in metrics.items():
                    if key != 'loss':
                        output[f"{metric_key_prefix}_{key}"] = value
                
                # Log the metrics
                if metrics:
                    self.log(output)
                
                # Save detailed results and metrics to files
                if self.args.output_dir:
                    import json
                    import os
                    
                    output_dir = self.args.output_dir
                    os.makedirs(output_dir, exist_ok=True)
                    
                    step_suffix = f"_step{self.state.global_step}" if self.state.global_step > 0 else ""
                    
                    # Save detailed results to JSONL file
                    if detailed_results is not None:
                        output_file = os.path.join(output_dir, f"{metric_key_prefix}_detailed{step_suffix}.jsonl")
                        with open(output_file, 'w') as f:
                            for result in detailed_results:
                                f.write(json.dumps(result) + '\n')
                        print(f"Detailed results saved to: {output_file}")
                    
                    # Save metrics to JSON file
                    if metrics:
                        metrics_file = os.path.join(output_dir, f"{metric_key_prefix}_metrics{step_suffix}.json")
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics, f, indent=2)
                        print(f"Metrics saved to: {metrics_file}")
                
            finally:
                # Clear stored predictions to free memory
                self._eval_predictions = []
                self._eval_labels = []
                self._eval_prompts = []
        
        return output

