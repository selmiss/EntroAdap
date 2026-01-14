from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    """
    Arguments for inference script.
    """
    input_file: str = field(
        default=None,
        metadata={"help": "Input file for inference (parquet or jsonl)."}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Output file for saving inference results."}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for inference."}
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens to generate."}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature (0 = greedy decoding)."}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Nucleus sampling threshold."}
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "Optional custom chat template to use."}
    )

@dataclass
class OctopusConfig:
    """
    Configuration for multi-modal Octopus model architecture used during inference.
    Contains model architecture parameters and runtime filtering options.
    """
    
    use_custom_model: bool = field(
        default=False,
        metadata={"help": "Whether to use the custom multi-modal model."}
    )
    
    modality_vocab_size: int = field(
        default=10000,
        metadata={"help": "Size of the modality vocabulary (backward compatible)."}
    )

    modality_embedding_dim: int = field(
        default=768,
        metadata={"help": "Dimension of modality embeddings."}
    )
    
    encoder_hidden_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden dimension for the graph encoder. If None, uses modality_embedding_dim."}
    )
    
    num_fusion_blocks: int = field(
        default=4,
        metadata={"help": "Number of fusion blocks."}
    )
    
    num_attention_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads in fusion blocks."}
    )
    
    fusion_hidden_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden dimension for fusion blocks. If None, uses modality_embedding_dim."}
    )
    
    fusion_intermediate_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Intermediate dimension for FFN in fusion blocks. If None, uses 4 * fusion_hidden_dim."}
    )
    
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate for fusion blocks."}
    )
    
    max_seq_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length for multimodal processing."}
    )

    # Patching (anchor/patch selection) parameters
    k_max: int = field(
        default=256,
        metadata={"help": "Maximum number of anchors/patches per graph."}
    )
    r_max: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum nodes kept per patch. Set to None to disable truncation."}
    )
    dynamic_k_mass: Optional[float] = field(
        default=0.1,
        metadata={"help": "Anchor selection mass threshold. Set to None for fixed top-k."}
    )
    beta: float = field(
        default=1.0,
        metadata={"help": "Distance scale for soft assignment (higher = sharper)."}
    )
    tau: float = field(
        default=0.1,
        metadata={"help": "Softmax temperature for assignment (lower = sharper)."}
    )
    
    max_atoms: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum atoms per structure. Structures exceeding this will be skipped. None means no limit."}
    )
    
    max_edges: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum edges per structure. Structures exceeding this will be skipped. None means no limit."}
    )
    
    skip_on_error: bool = field(
        default=True,
        metadata={"help": "Skip samples that fail to load or exceed thresholds."}
    )
    
    octopus_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trained Octopus checkpoint directory for loading the full model."}
    )