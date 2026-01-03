"""
Tests for multi-dataset loading and modality-aware batch sampling.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from src.data_loader.aa_dataset import GraphDataset, ModalityAwareBatchSampler


def test_graphdataset_accepts_single_path():
    """Test that GraphDataset still works with single path (backward compatibility)."""
    # This test would require actual data files, so we'll use mock
    with patch('src.data_loader.aa_dataset.load_dataset') as mock_load:
        # Mock the dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load.return_value = {'train': mock_dataset}
        
        dataset = GraphDataset(dataset_path="data/test.parquet")
        assert dataset.dataset == mock_dataset


def test_graphdataset_accepts_comma_separated_paths():
    """Test that GraphDataset accepts comma-separated paths."""
    with patch('src.data_loader.aa_dataset.load_dataset') as mock_load:
        with patch('src.data_loader.aa_dataset.concatenate_datasets') as mock_concat:
            # Mock datasets
            mock_ds1 = Mock()
            mock_ds1.__len__ = Mock(return_value=50)
            mock_ds2 = Mock()
            mock_ds2.__len__ = Mock(return_value=50)
            mock_load.return_value = {'train': mock_ds1}
            
            # Mock concatenate to return a combined dataset
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=100)
            mock_concat.return_value = mock_combined
            
            dataset = GraphDataset(dataset_path="data/test1.parquet,data/test2.parquet")
            
            # Should have called load_dataset twice
            assert mock_load.call_count == 2
            # Should have called concatenate_datasets once
            assert mock_concat.call_count == 1


def test_graphdataset_accepts_list_of_paths():
    """Test that GraphDataset accepts list of paths."""
    with patch('src.data_loader.aa_dataset.load_dataset') as mock_load:
        with patch('src.data_loader.aa_dataset.concatenate_datasets') as mock_concat:
            # Mock datasets
            mock_ds = Mock()
            mock_ds.__len__ = Mock(return_value=50)
            mock_load.return_value = {'train': mock_ds}
            
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=150)
            mock_concat.return_value = mock_combined
            
            dataset = GraphDataset(
                dataset_path=["data/test1.parquet", "data/test2.parquet", "data/test3.parquet"]
            )
            
            # Should have called load_dataset three times
            assert mock_load.call_count == 3
            # Should have called concatenate_datasets once
            assert mock_concat.call_count == 1


def test_modality_aware_batch_sampler_groups_by_modality():
    """Test that ModalityAwareBatchSampler correctly groups samples by modality."""
    # Create a mock dataset
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)
    
    # Mock __getitem__ to return different modalities
    def mock_getitem(idx):
        if idx < 30:
            return {'modality': 'protein'}
        elif idx < 60:
            return {'modality': 'molecule'}
        elif idx < 80:
            return {'modality': 'dna'}
        else:
            return {'modality': 'rna'}
    
    mock_dataset.__getitem__ = Mock(side_effect=mock_getitem)
    
    # Create sampler
    sampler = ModalityAwareBatchSampler(
        dataset=mock_dataset,
        batch_size=10,
        shuffle=False,
        seed=42,
    )
    
    # Check modality grouping
    assert len(sampler.modality_indices) == 4
    assert len(sampler.modality_indices['protein']) == 30
    assert len(sampler.modality_indices['molecule']) == 30
    assert len(sampler.modality_indices['dna']) == 20
    assert len(sampler.modality_indices['rna']) == 20


def test_modality_aware_batch_sampler_creates_same_modality_batches():
    """Test that each batch contains only samples from the same modality."""
    # Create a mock dataset
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=60)
    
    # Assign modalities: 0-19 protein, 20-39 molecule, 40-59 dna
    def mock_getitem(idx):
        if idx < 20:
            return {'modality': 'protein'}
        elif idx < 40:
            return {'modality': 'molecule'}
        else:
            return {'modality': 'dna'}
    
    mock_dataset.__getitem__ = Mock(side_effect=mock_getitem)
    
    # Create sampler
    sampler = ModalityAwareBatchSampler(
        dataset=mock_dataset,
        batch_size=10,
        shuffle=False,
        seed=42,
    )
    
    # Iterate through batches and verify same-modality constraint
    for batch_indices in sampler:
        # Get modalities for this batch
        modalities = set(mock_getitem(idx)['modality'] for idx in batch_indices)
        # All samples in batch should have the same modality
        assert len(modalities) == 1, f"Batch has mixed modalities: {modalities}"


def test_modality_aware_batch_sampler_respects_batch_size():
    """Test that ModalityAwareBatchSampler respects the batch size."""
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=100)
    
    def mock_getitem(idx):
        return {'modality': 'protein'}  # All same modality for simplicity
    
    mock_dataset.__getitem__ = Mock(side_effect=mock_getitem)
    
    batch_size = 16
    sampler = ModalityAwareBatchSampler(
        dataset=mock_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
        drop_last=True,  # Drop incomplete batches
    )
    
    # Check all batches have the correct size
    for batch_indices in sampler:
        assert len(batch_indices) == batch_size


def test_modality_aware_batch_sampler_set_epoch():
    """Test that set_epoch changes the random seed for shuffling."""
    mock_dataset = Mock()
    mock_dataset.__len__ = Mock(return_value=50)
    
    def mock_getitem(idx):
        return {'modality': 'protein'}
    
    mock_dataset.__getitem__ = Mock(side_effect=mock_getitem)
    
    sampler = ModalityAwareBatchSampler(
        dataset=mock_dataset,
        batch_size=10,
        shuffle=True,
        seed=42,
    )
    
    # Get batches for epoch 0
    epoch0_batches = list(sampler)
    
    # Set epoch to 1
    sampler.set_epoch(1)
    epoch1_batches = list(sampler)
    
    # Batches should be different due to different shuffling
    assert epoch0_batches != epoch1_batches


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

