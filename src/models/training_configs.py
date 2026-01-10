# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import trl


@dataclass
class OctopusConfig:
    """
    Configuration for multi-modal model architecture.
    
    Args:
        use_custom_model (`bool`, *optional*, defaults to `False`):
            Whether to use the custom multi-modal model instead of standard LLM.
        modality_vocab_size (`int`, *optional*, defaults to 10000):
            Size of the modality vocabulary (kept for backward compatibility; may be unused by some models).
        modality_embedding_dim (`int`, *optional*, defaults to 768):
            Dimension of modality embeddings.
        num_fusion_blocks (`int`, *optional*, defaults to 4):
            Number of fusion blocks to apply.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in fusion blocks.
        fusion_hidden_dim (`int`, *optional*, defaults to `None`):
            Hidden dimension for fusion blocks. If None, uses modality_embedding_dim.
        fusion_intermediate_dim (`int`, *optional*, defaults to `None`):
            Intermediate dimension for FFN in fusion blocks. If None, uses 4 * fusion_hidden_dim.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate for fusion blocks.
        max_atoms (`int`, *optional*, defaults to `None`):
            Maximum atoms per structure. Structures exceeding this will be skipped at runtime.
        max_edges (`int`, *optional*, defaults to `None`):
            Maximum edges per structure. Structures exceeding this will be skipped at runtime.
        skip_on_error (`bool`, *optional*, defaults to `True`):
            Skip samples that fail to load or exceed thresholds instead of raising errors.
    """
    
    use_custom_model: bool = field(
        default=False,
        metadata={"help": "Whether to use the custom multi-modal model instead of standard LLM."}
    )
    modality_vocab_size: int = field(
        default=10000,
        metadata={
            "help": "Size of the modality vocabulary (backward compatible; may be unused by some models)."
        },
    )
    modality_embedding_dim: int = field(
        default=768,
        metadata={"help": "Dimension of modality embeddings (used for fusion blocks and projections)."}
    )
    encoder_hidden_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden dimension for the graph encoder. If None, uses modality_embedding_dim. Set this to match your pretrained encoder checkpoint (e.g., 256) when loading from a checkpoint with different dimension."}
    )
    num_fusion_blocks: int = field(
        default=4,
        metadata={"help": "Number of fusion blocks to apply."}
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
        default=2048,
        metadata={"help": "Maximum sequence length for multimodal training."}
    )
    
    # Runtime filtering parameters
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
        metadata={"help": "Skip samples that fail to load or exceed thresholds instead of raising errors."}
    )
    
    # Encoder checkpoint for initialization
    encoder_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained encoder checkpoint directory (e.g., checkpoints/aa_encoder_multi_limited). If provided, the encoder weights will be loaded from this checkpoint before training."}
    )
    
    # Full Octopus model checkpoint for loading trained models
    octopus_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a trained Octopus checkpoint directory with sharded weights (e.g., checkpoints/octopus/stage1). If provided, loads the full Octopus model (LLM + encoder + fusion blocks + gates) before training. Use this for stage-based training with different settings (e.g., unfreezing LLM, adding LoRA)."}
    )
    prepared_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a prepared Octopus checkpoint directory. Uses config-first loading (from_config + load_state_dict). Alternative to octopus_checkpoint_path."}
    )
    
    # Freezing options for training
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the graph encoder during training."}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the LLM during training."}
    )
    freeze_gates: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the anchor and edge gates during training."}
    )
    freeze_fusion_blocks: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the fusion blocks during training."}
    )
    freeze_projections: bool = field(
        default=False,
        metadata={"help": "Whether to freeze all projection layers (instr_proj, patch_proj, node_proj, output_proj) during training."}
    )


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    # Override the dataset_name to make it optional
    # Note: This will be converted to support lists in __post_init__
    dataset_name: Optional[Any] = field(
        default=None, 
        metadata={"help": "Dataset name or list of dataset paths (files or directories). Each directory will load all parquet files. Can be omitted if using dataset_mixture."}
    )
    dataset_train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to training file. Use with dataset_eval_file for pre-split datasets."}
    )
    dataset_eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation file. Use with dataset_train_file for pre-split datasets."}
    )
    dataset_train_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum samples to load from training file. None means no limit."}
    )
    dataset_eval_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum samples to load from evaluation file. None means no limit."}
    )
    dataset_max_samples: Optional[Any] = field(
        default=None,
        metadata={"help": "Maximum samples per dataset. Can be a single integer (applied to all) or a list of integers matching dataset_name length. None means no limit."}
    )
    eval_split_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Ratio of data to use for evaluation (e.g., 0.1 for 10%). If provided, the training data will be split into train and eval sets."}
    )
    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures with advanced options like shuffling."},
    )

    def __post_init__(self):
        if self.dataset_train_file is not None or self.dataset_eval_file is not None:
            if self.dataset_train_file is None or self.dataset_eval_file is None:
                raise ValueError("Both dataset_train_file and dataset_eval_file must be specified together")
            if self.dataset_name is not None or self.dataset_mixture is not None:
                raise ValueError("Cannot use dataset_train_file/dataset_eval_file with dataset_name or dataset_mixture")
        
        if self.dataset_name is not None:
            if not isinstance(self.dataset_name, (str, list)):
                raise ValueError(
                    f"dataset_name must be a string or list, got {type(self.dataset_name)}"
                )
            if isinstance(self.dataset_name, list):
                # Validate all items are strings
                if not all(isinstance(item, str) for item in self.dataset_name):
                    raise ValueError(
                        "All items in dataset_name list must be strings (file or directory paths)"
                    )
        
        # Validate dataset_max_samples
        if self.dataset_max_samples is not None:
            if isinstance(self.dataset_max_samples, (int, float)):
                # Single value - will be applied to all datasets
                pass
            elif isinstance(self.dataset_max_samples, list):
                # List of values - must match dataset_name length
                if self.dataset_name is None:
                    raise ValueError("dataset_max_samples list provided but dataset_name is None")
                if isinstance(self.dataset_name, list):
                    if len(self.dataset_max_samples) != len(self.dataset_name):
                        raise ValueError(
                            f"dataset_max_samples list length ({len(self.dataset_max_samples)}) "
                            f"must match dataset_name list length ({len(self.dataset_name)})"
                        )
                # Validate all items are None or integers
                if not all(item is None or isinstance(item, (int, float)) for item in self.dataset_max_samples):
                    raise ValueError(
                        "All items in dataset_max_samples list must be None or integers"
                    )
            else:
                raise ValueError(
                    f"dataset_max_samples must be an integer, a list of integers, or None, got {type(self.dataset_max_samples)}"
                )
        
        # Validate eval_split_ratio
        if self.eval_split_ratio is not None:
            if not isinstance(self.eval_split_ratio, (int, float)):
                raise ValueError(
                    f"eval_split_ratio must be a number between 0 and 1, got {type(self.eval_split_ratio)}"
                )
            if not 0 < self.eval_split_ratio < 1:
                raise ValueError(
                    f"eval_split_ratio must be between 0 and 1, got {self.eval_split_ratio}"
                )
        
        if self.dataset_name is None and self.dataset_mixture is None and self.dataset_train_file is None:
            raise ValueError("Either `dataset_name`, `dataset_mixture`, or `dataset_train_file/dataset_eval_file` must be provided")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    num_completions_to_print: int = field(default=0, metadata={"help": "Number of completions to print."})
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log the unique prompts to wandb. This will create a new run for each unique prompt.")
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation."},
    )
    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    eval_metrics: str = field(
        default="none",
        metadata={"help": "Evaluation metrics to compute: 'text' for NLP metrics (BLEU, ROUGE, METEOR), 'qa' for multiple choice accuracy, 'none' to disable."},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum number of new tokens to generate during evaluation. If None, uses model's default generation config."},
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'ioi_code', 'code_format', 'soft_overlong_punishment'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
        max_completion_len (`int`):
            Maximum number of tokens in completion.
        soft_punish_cache (`int`):
            Minimum number of tokens in completion.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        # '(?:python|cpp)'
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
    code_eval_scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = field(
        default="weighted_sum",
        metadata={"help": "use fraction of passed test cases as reward. If false, use 0/1 scoring."},
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": "Number of parallel E2B code executions per process. Default of 2 is suitable for the Free Hobby tier of E2B with 8 GPUs used for training."
        },
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )

    e2b_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the E2B router. See scripts/e2b_router.py"},
    )

    morph_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the MorphCloud router. See scripts/morph_router.py"},
    )

    code_provider: Optional[str] = field(
        default="e2b",
        metadata={
            "help": "Provider for code execution. Options: 'e2b', 'local', 'morph'.",
            "choices": ["e2b", "local", "morph"],
        },
    )

    ioi_provider: Optional[str] = field(
        default="piston",
        metadata={
            "help": "Provider for IOI code execution. Options: 'piston', 'morph'.",
            "choices": ["piston", "morph"],
        },
    )

    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )
