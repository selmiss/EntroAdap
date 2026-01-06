import logging
import os
from typing import List, Dict
from collections import Counter

import datasets
from datasets import DatasetDict, concatenate_datasets

from src.models.training_configs import ScriptArguments


logger = logging.getLogger(__name__)


def count_modality_samples(dataset: datasets.Dataset) -> Dict[str, int]:
    """
    Count the number of samples for each modality in the dataset.
    
    Args:
        dataset: HuggingFace dataset with 'modality' column
        
    Returns:
        Dictionary mapping modality names to sample counts
    """
    if 'modality' not in dataset.column_names:
        logger.warning("Dataset does not have 'modality' column. Cannot count modalities.")
        return {}
    
    # Use Counter for efficient counting
    modality_counts = Counter(dataset['modality'])
    
    return dict(modality_counts)


def print_modality_statistics(dataset: datasets.Dataset, dataset_name: str = "Dataset"):
    """
    Print modality statistics in a formatted table.
    
    Args:
        dataset: HuggingFace dataset with 'modality' column
        dataset_name: Name to display for the dataset
    """
    modality_counts = count_modality_samples(dataset)
    
    if not modality_counts:
        logger.info(f"{dataset_name}: No modality information available")
        return
    
    total = sum(modality_counts.values())
    
    logger.info(f"\n{'='*80}")
    logger.info(f"{dataset_name} - Modality Statistics:")
    logger.info(f"{'='*80}")
    logger.info(f"{'Modality':<20} {'Count':>15} {'Percentage':>15}")
    logger.info(f"{'-'*80}")
    
    for modality in sorted(modality_counts.keys()):
        count = modality_counts[modality]
        percentage = (count / total) * 100
        logger.info(f"{modality:<20} {count:>15,} {percentage:>14.2f}%")
    
    logger.info(f"{'-'*80}")
    logger.info(f"{'Total':<20} {total:>15,} {100.0:>14.2f}%")
    logger.info(f"{'='*80}\n")


def _load_parquet_files_from_paths(paths: List[str], max_samples_list: List = None) -> List:
    """
    Load parquet files from a list of paths.
    Each path can be:
    - A directory: loads all .parquet files recursively
    - A file: loads that specific file
    
    Args:
        paths: List of file or directory paths
        max_samples_list: Optional list of max samples per dataset. None means no limit.
        
    Returns:
        List of loaded datasets
    """
    datasets_list = []
    all_parquet_files = []
    
    for path in paths:
        if os.path.isdir(path):
            # Find all parquet files in the directory (including subdirectories)
            logger.info(f"Scanning directory: {path}")
            dir_parquet_files = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.parquet'):
                        dir_parquet_files.append(os.path.join(root, file))
            
            if not dir_parquet_files:
                logger.warning(f"No parquet files found in directory: {path}")
            else:
                logger.info(f"  Found {len(dir_parquet_files)} parquet files in {path}")
                all_parquet_files.extend(dir_parquet_files)
        elif os.path.isfile(path):
            # Single file
            logger.info(f"Adding file: {path}")
            all_parquet_files.append(path)
        else:
            logger.warning(f"Path does not exist or is not accessible: {path}")
    
    if not all_parquet_files:
        raise ValueError(f"No parquet files found in provided paths: {paths}")
    
    logger.info(f"\nLoading {len(all_parquet_files)} total parquet files:")
    for pf in all_parquet_files:
        logger.info(f"  - {pf}")
    
    # If max_samples_list is provided, ensure it matches
    if max_samples_list is not None:
        if len(max_samples_list) != len(all_parquet_files):
            # If it's shorter, extend with None
            max_samples_list = list(max_samples_list) + [None] * (len(all_parquet_files) - len(max_samples_list))
    else:
        max_samples_list = [None] * len(all_parquet_files)
    
    # Load all parquet files
    for idx, (parquet_file, max_samples) in enumerate(zip(all_parquet_files, max_samples_list)):
        ds = datasets.load_dataset('parquet', data_files=parquet_file)
        # Extract the actual dataset from DatasetDict
        if 'train' in ds:
            ds = ds['train']
        else:
            ds = ds[list(ds.keys())[0]]
        
        original_len = len(ds)
        
        # Apply max_samples if specified
        if max_samples is not None and max_samples < original_len:
            import random
            random.seed(42)
            indices = list(range(original_len))
            random.shuffle(indices)
            selected_indices = indices[:int(max_samples)]
            ds = ds.select(selected_indices)
            logger.info(f"  [{idx+1}] {parquet_file}: {original_len:,} -> {len(ds):,} samples (max_samples={int(max_samples):,})")
        else:
            logger.info(f"  [{idx+1}] {parquet_file}: {original_len:,} samples")
        
        datasets_list.append(ds)
    
    return datasets_list


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        
        # Process max_samples parameter
        max_samples_list = None
        if args.dataset_max_samples is not None:
            if isinstance(args.dataset_max_samples, (int, float)):
                # Single value applied to all datasets
                if isinstance(args.dataset_name, list):
                    max_samples_list = [args.dataset_max_samples] * len(args.dataset_name)
                else:
                    max_samples_list = [args.dataset_max_samples]
            elif isinstance(args.dataset_max_samples, list):
                # List of values
                max_samples_list = args.dataset_max_samples
        
        # Check if dataset_name is a list
        if isinstance(args.dataset_name, list):
            logger.info(f"Processing list of {len(args.dataset_name)} dataset paths")
            datasets_list = _load_parquet_files_from_paths(args.dataset_name, max_samples_list)
            
            # Concatenate all datasets
            combined_dataset = concatenate_datasets(datasets_list)
            logger.info(f"Loaded total of {len(combined_dataset)} examples from {len(datasets_list)} files")
            
            # Apply train/eval split if requested
            if args.eval_split_ratio is not None:
                logger.info(f"Splitting dataset with eval_split_ratio={args.eval_split_ratio}")
                split_dataset = combined_dataset.train_test_split(
                    test_size=args.eval_split_ratio,
                    seed=42,
                )
                logger.info(f"  Train: {len(split_dataset['train']):,} samples")
                logger.info(f"  Eval:  {len(split_dataset['test']):,} samples")
                return DatasetDict({"train": split_dataset['train'], "test": split_dataset['test']})
            else:
                return DatasetDict({"train": combined_dataset})
        
        # Check if dataset_name is a directory
        elif os.path.isdir(args.dataset_name):
            logger.info(f"Detected directory: {args.dataset_name}")
            datasets_list = _load_parquet_files_from_paths([args.dataset_name], max_samples_list)
            
            # Concatenate all datasets
            combined_dataset = concatenate_datasets(datasets_list)
            logger.info(f"Loaded total of {len(combined_dataset)} examples from {len(datasets_list)} files")
            
            # Apply train/eval split if requested
            if args.eval_split_ratio is not None:
                logger.info(f"Splitting dataset with eval_split_ratio={args.eval_split_ratio}")
                split_dataset = combined_dataset.train_test_split(
                    test_size=args.eval_split_ratio,
                    seed=42,
                )
                logger.info(f"  Train: {len(split_dataset['train']):,} samples")
                logger.info(f"  Eval:  {len(split_dataset['test']):,} samples")
                return DatasetDict({"train": split_dataset['train'], "test": split_dataset['test']})
            else:
                return DatasetDict({"train": combined_dataset})
        
        # Check if dataset_name is a local file
        elif os.path.exists(args.dataset_name) or args.dataset_name.endswith(('.json', '.jsonl', '.csv', '.parquet', '.txt')):
            logger.info(f"Detected local file: {args.dataset_name}")
            # For local files, infer the format from extension
            if args.dataset_name.endswith('.jsonl') or args.dataset_name.endswith('.json'):
                ds = datasets.load_dataset('json', data_files=args.dataset_name)
            elif args.dataset_name.endswith('.csv'):
                ds = datasets.load_dataset('csv', data_files=args.dataset_name)
            elif args.dataset_name.endswith('.parquet'):
                ds = datasets.load_dataset('parquet', data_files=args.dataset_name)
            else:
                ds = datasets.load_dataset('text', data_files=args.dataset_name)
            
            # Apply max_samples if specified
            if max_samples_list and max_samples_list[0] is not None:
                if 'train' in ds:
                    original_len = len(ds['train'])
                    if max_samples_list[0] < original_len:
                        import random
                        random.seed(42)
                        indices = list(range(original_len))
                        random.shuffle(indices)
                        selected_indices = indices[:int(max_samples_list[0])]
                        ds['train'] = ds['train'].select(selected_indices)
                        logger.info(f"Applied max_samples: {original_len:,} -> {len(ds['train']):,} samples")
            
            # Apply train/eval split if requested
            if args.eval_split_ratio is not None and 'train' in ds:
                logger.info(f"Splitting dataset with eval_split_ratio={args.eval_split_ratio}")
                split_dataset = ds['train'].train_test_split(
                    test_size=args.eval_split_ratio,
                    seed=42,
                )
                logger.info(f"  Train: {len(split_dataset['train']):,} samples")
                logger.info(f"  Eval:  {len(split_dataset['test']):,} samples")
                return DatasetDict({"train": split_dataset['train'], "test": split_dataset['test']})
            
            return ds
        else:
            # Regular HuggingFace Hub dataset
            return datasets.load_dataset(args.dataset_name, args.dataset_config)
    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                return combined_dataset
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
