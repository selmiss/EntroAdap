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

from src.models.training_configs import ScriptArguments, SFTConfig, OctopusConfig
from src.data_loader import MultiModalDataCollator, preprocess_multimodal_dataset
from src.trainer.octopus_trainer import MultiModalSFTTrainer
from utils import get_dataset, get_tokenizer
from utils.env_utils import expand_env_vars
from utils.model_utils import get_model_and_peft_config
from utils.callbacks import get_callbacks
from utils.wandb_logging import init_wandb_training
from trl import ModelConfig, SFTTrainer, TrlParser


logger = logging.getLogger(__name__)


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
    # Set transformers to WARNING to reduce verbose config output
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # logger.info(f"Model parameters {model_args}")
    # logger.info(f"Script parameters {script_args}")
    # logger.info(f"Training parameters {training_args}")
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
    
    # Load model + PEFT config (standard LLM vs custom multimodal)
    model, peft_config = get_model_and_peft_config(model_args, training_args, multimodal_args)

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
    
    # Add multimodal special token for graph/structure injection
    if multimodal_args is not None and multimodal_args.use_custom_model:
        structure_token = "<STRUCTURE>"
        if structure_token not in tokenizer.get_vocab():
            logger.info(f"Adding special token: {structure_token}")
            tokenizer.add_special_tokens({"additional_special_tokens": [structure_token]})
        
        # Resize model embeddings if new tokens were added
        # NOTE: For the custom multimodal wrapper, resize the *inner* LLM embeddings.
        inner_llm = getattr(model, "llm_model", None)
        (inner_llm or model).resize_token_embeddings(len(tokenizer))

    ############################
    # Initialize the SFT Trainer
    ############################
    # Use custom data collator for multimodal training
    data_collator = None
    use_multimodal = multimodal_args is not None and multimodal_args.use_custom_model
    
    if use_multimodal:
        logger.info("Using custom MultiModalDataCollator and MultiModalSFTTrainer for multimodal training")

        data_collator = MultiModalDataCollator(
            tokenizer=tokenizer,
            padding=True,
            max_length=multimodal_args.max_seq_length,
            return_tensors="pt",
        )
        dataset = preprocess_multimodal_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            split=script_args.dataset_train_split,
            max_seq_length=multimodal_args.max_seq_length,
        )
    
    # Use custom trainer for multimodal, standard for text-only
    trainer_class = MultiModalSFTTrainer if use_multimodal else SFTTrainer
    trainer = trainer_class(
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
    # Always parse with OctopusConfig; treat it as "enabled" only when use_custom_model=True.
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, OctopusConfig))
    script_args, training_args, model_args, multimodal_args = parser.parse_args_and_config()

    # Keep downstream behavior consistent: only pass multimodal_args when actually used.
    if not getattr(multimodal_args, "use_custom_model", False):
        multimodal_args = None

    expand_env_vars(script_args, training_args, model_args, multimodal_args)
    main(script_args, training_args, model_args, multimodal_args)