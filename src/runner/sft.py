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

"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.configs import ScriptArguments, SFTConfig, MultiModalConfig
from src.data_loader import MultiModalDataCollator, preprocess_multimodal_dataset
from utils import get_dataset, get_model, get_tokenizer
from utils.model_utils import get_custom_model
from utils.callbacks import get_callbacks
from utils.wandb_logging import init_wandb_training
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


def expand_env_vars(script_args, training_args):
    """Expand environment variables in paths (e.g., ${CHECKPOINT_DIR}, ${DATA_DIR})."""
    training_args.output_dir = os.path.expandvars(training_args.output_dir)
    if script_args.dataset_name:
        script_args.dataset_name = os.path.expandvars(script_args.dataset_name)
    if script_args.dataset_mixture:
        script_args.dataset_mixture = {
            os.path.expandvars(k): v for k, v in script_args.dataset_mixture.items()
        }


def main(script_args, training_args, model_args, multimodal_args=None):
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")
    if multimodal_args is not None:
        logger.info(f"Multi-modal parameters {multimodal_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    dataset = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)
    
    # Load model (custom multimodal or standard LLM)
    if multimodal_args is not None and multimodal_args.use_custom_model:
        logger.info("Loading custom multi-modal model...")
        model = get_custom_model(model_args, training_args, multimodal_args)
        # For custom models, PEFT and liger_kernel are already applied inside get_custom_model to the inner LLM
        peft_config = None
    else:
        logger.info("Loading standard LLM model...")
        model = get_model(model_args, training_args)
        # For standard models, let the trainer handle PEFT
        peft_config = get_peft_config(model_args)

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        # Manually set ChatML template since setup_chat_format was removed in TRL 0.26+
        CHATML_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = CHATML_TEMPLATE
        # Add special tokens if needed
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Resize model embeddings if new tokens were added
        model.resize_token_embeddings(len(tokenizer))

    ############################
    # Initialize the SFT Trainer
    ############################
    # Use custom data collator for multimodal training
    data_collator = None
    if multimodal_args is not None and multimodal_args.use_custom_model:
        
        logger.info("Using custom MultiModalDataCollator for multimodal training")
        # Default to 256 for multimodal (common short-context molecule tasks)
        max_seq_length = 256
        data_collator = MultiModalDataCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        
        # Preprocess dataset: tokenize messages and create labels
        dataset = preprocess_multimodal_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            split=script_args.dataset_train_split,
            max_seq_length=max_seq_length,
        )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        data_collator=data_collator,
        peft_config=peft_config,
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    # Try to parse with MultiModalConfig, fall back to without if not needed
    try:
        parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, MultiModalConfig))
        script_args, training_args, model_args, multimodal_args = parser.parse_args_and_config()
        expand_env_vars(script_args, training_args)
        main(script_args, training_args, model_args, multimodal_args)
    except (ValueError, TypeError):
        # If MultiModalConfig is not in the config, use standard parsing
        parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
        script_args, training_args, model_args = parser.parse_args_and_config()
        expand_env_vars(script_args, training_args)
        main(script_args, training_args, model_args, multimodal_args=None)