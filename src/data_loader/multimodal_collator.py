# Multi-modal data collator for training with text and modality tokens

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy
from datasets import Dataset, DatasetDict


@dataclass
class MultiModalDataCollator:
    """
    Collates batches with text and optional modality tokens.
    
    Works with SFTTrainer by handling both pre-tokenized inputs
    (input_ids, labels) and adding multimodal fields.
    
    Note: This collator now expects pre-computed embeddings instead of tokens.
    The fields should be:
    - modality_embeddings: List of embeddings (each is a list of floats)
    - kv_embeddings: List of embeddings (each is a list of floats)
    """
    
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    include_modality_tokens: bool = True
    
    def _pad_text_features(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pad text sequences and return batch."""
        # Extract text features (without labels - tokenizer doesn't handle labels)
        text_features = []
        has_labels = "labels" in features[0]
        
        for f in features:
            text_feat = {
                "input_ids": f["input_ids"],
                "attention_mask": f.get("attention_mask", [1] * len(f["input_ids"])),
            }
            text_features.append(text_feat)
        
        # Use tokenizer to pad input_ids and attention_mask
        batch = self.tokenizer.pad(
            text_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Manually pad labels if present (tokenizer can't handle this)
        if has_labels:
            labels = [f["labels"] for f in features]
            max_len = batch["input_ids"].shape[1]
            padded_labels = []
            for label in labels:
                pad_len = max_len - len(label)
                padded_labels.append(label + [-100] * pad_len)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch
    
    def _add_modality_embeddings(self, batch: Dict[str, Any], features: List[Dict[str, Any]]) -> None:
        """Add padded modality embeddings to batch."""
        if not self.include_modality_tokens or not any("modality_embeddings" in f for f in features):
            return
        
        modality_embeddings_list = [f.get("modality_embeddings", []) for f in features]
        if not any(modality_embeddings_list):
            return
        
        # Get max sequence length and embedding dimension
        max_len = max(len(e) for e in modality_embeddings_list if e)
        if max_len == 0:
            return
        
        # Get embedding dimension from first non-empty example
        embed_dim = None
        for emb in modality_embeddings_list:
            if emb and len(emb) > 0:
                embed_dim = len(emb[0])
                break
        
        if embed_dim is None:
            return
        
        padded_embeddings, masks = [], []
        
        for embeddings in modality_embeddings_list:
            if embeddings and len(embeddings) > 0:
                pad_len = max_len - len(embeddings)
                # Pad with zero embeddings
                padding = [[0.0] * embed_dim] * pad_len
                padded_embeddings.append(embeddings + padding)
                masks.append([1] * len(embeddings) + [0] * pad_len)
            else:
                padded_embeddings.append([[0.0] * embed_dim] * max_len)
                masks.append([0] * max_len)
        
        batch["modality_embeddings"] = torch.tensor(padded_embeddings, dtype=torch.float32)
        batch["modality_attention_mask"] = torch.tensor(masks, dtype=torch.long)
    
    def _add_kv_embeddings(self, batch: Dict[str, Any], features: List[Dict[str, Any]]) -> None:
        """Add padded cross-attention text embeddings to batch."""
        if not any("kv_embeddings" in f for f in features):
            return
        
        embeddings_list = [f.get("kv_embeddings", []) for f in features]
        if not any(embeddings_list):
            return
        
        # Get max sequence length and embedding dimension
        max_len = max(len(e) for e in embeddings_list if e)
        if max_len == 0:
            return
        
        # Get embedding dimension from first non-empty example
        embed_dim = None
        for emb in embeddings_list:
            if emb and len(emb) > 0:
                embed_dim = len(emb[0])
                break
        
        if embed_dim is None:
            return
        
        padded_embeddings, masks = [], []
        
        for embeddings in embeddings_list:
            if embeddings and len(embeddings) > 0:
                pad_len = max_len - len(embeddings)
                # Pad with zero embeddings
                padding = [[0.0] * embed_dim] * pad_len
                padded_embeddings.append(embeddings + padding)
                masks.append([1] * len(embeddings) + [0] * pad_len)
            else:
                padded_embeddings.append([[0.0] * embed_dim] * max_len)
                masks.append([0] * max_len)
        
        batch["kv_embeddings"] = torch.tensor(padded_embeddings, dtype=torch.float32)
        batch["text_attention_mask"] = torch.tensor(masks, dtype=torch.long)
    
    def _add_modality_positions(self, batch: Dict[str, Any], features: List[Dict[str, Any]]) -> None:
        """Add padded modality positions to batch."""
        if not any("modality_positions" in f for f in features):
            return
        
        positions_list = [f.get("modality_positions", []) for f in features]
        if not any(positions_list):
            return
        
        max_len = max(len(p) for p in positions_list if p)
        padded_positions = []
        for pos in positions_list:
            if pos:
                padded_positions.append(pos + [-1] * (max_len - len(pos)))
            else:
                padded_positions.append([-1] * max_len)
        batch["modality_positions"] = torch.tensor(padded_positions, dtype=torch.long)
    
    def _validate_embeddings(self, features: List[Dict[str, Any]]) -> None:
        """Validate that embeddings are present if multimodal mode is enabled."""
        # This is a placeholder for any validation logic needed
        # Could add checks that embeddings have the right shape, dtype, etc.
        pass
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of examples with text and optional modality embeddings."""
        # Validate embeddings if present
        self._validate_embeddings(features)
        
        # Pad text features (input_ids, labels, attention_mask)
        batch = self._pad_text_features(features)
        
        # Add multimodal fields (now using embeddings instead of tokens)
        self._add_modality_embeddings(batch, features)
        self._add_kv_embeddings(batch, features)
        self._add_modality_positions(batch, features)
        
        return batch


def preprocess_multimodal_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    max_seq_length: int = 1024,
) -> DatasetDict:
    """
    Preprocess multimodal dataset by tokenizing messages and creating labels.
    
    This function:
    1. Applies chat template to convert messages to text
    2. Tokenizes the text into input_ids
    3. Creates labels (copies of input_ids for causal LM training)
    4. Preserves multimodal fields (modality_tokens, cross_attention_text, etc.)
    
    Args:
        dataset: Dataset dictionary containing the data
        tokenizer: Tokenizer to use for processing
        split: Split name to process (e.g., "train")
        max_seq_length: Maximum sequence length for truncation
    
    Returns:
        Processed dataset with tokenized inputs and labels
    """
    def _preprocess_batch(examples):
        """Tokenize messages and prepare labels while preserving multimodal fields."""
        # Apply chat template to messages
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # Padding handled by collator
        )
        
        # For causal LM, labels = input_ids (copy each list in the batch)
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        
        return tokenized
    
    # Apply preprocessing to dataset
    processed_dataset = dataset.map(
        _preprocess_batch,
        batched=True,
        desc="Tokenizing dataset",
        remove_columns=["messages"] if "messages" in dataset[split].column_names else [],
    )
    
    return processed_dataset
