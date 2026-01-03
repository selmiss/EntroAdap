"""
Tests for enhanced multi-dataset features including max samples and stratified splits.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from src.data_loader.aa_dataset import GraphDataset, ModalityAwareBatchSampler


def test_graphdataset_with_max_samples():
    """Test that GraphDataset respects max_samples_per_dataset."""
    with patch('src.data_loader.aa_dataset.load_dataset') as mock_load:
        with patch('src.data_loader.aa_dataset.concatenate_datasets') as mock_concat:
            # Mock datasets with select method (new implementation doesn't use shuffle)
            mock_ds1 = MagicMock()
            mock_ds1.__len__ = Mock(return_value=1000)
            selected1 = MagicMock()
            selected1.__len__ = Mock(return_value=100)
            mock_ds1.select = Mock(return_value=selected1)
            
            mock_ds2 = MagicMock()
            mock_ds2.__len__ = Mock(return_value=500)
            selected2 = MagicMock()
            selected2.__len__ = Mock(return_value=200)
            mock_ds2.select = Mock(return_value=selected2)
            
            mock_load.side_effect = [{'train': mock_ds1}, {'train': mock_ds2}]
            
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=300)
            mock_concat.return_value = mock_combined
            
            # Test with list of max samples
            dataset = GraphDataset(
                dataset_path=["data/test1.parquet", "data/test2.parquet"],
                max_samples_per_dataset=[100, 200]
            )
            
            # Should have selected from both datasets
            mock_ds1.select.assert_called_once()
            mock_ds2.select.assert_called_once()
            
            # Verify correct number of samples were selected
            call_args1 = mock_ds1.select.call_args[0][0]
            call_args2 = mock_ds2.select.call_args[0][0]
            assert len(call_args1) == 100  # First 100 indices
            assert len(call_args2) == 200  # First 200 indices


def test_graphdataset_with_single_max_samples():
    """Test that GraphDataset applies single max_samples to all datasets."""
    with patch('src.data_loader.aa_dataset.load_dataset') as mock_load:
        with patch('src.data_loader.aa_dataset.concatenate_datasets') as mock_concat:
            # Mock datasets
            mock_ds1 = MagicMock()
            mock_ds1.__len__ = Mock(return_value=1000)
            selected1 = MagicMock()
            selected1.__len__ = Mock(return_value=500)
            mock_ds1.select = Mock(return_value=selected1)
            
            mock_ds2 = MagicMock()
            mock_ds2.__len__ = Mock(return_value=800)
            selected2 = MagicMock()
            selected2.__len__ = Mock(return_value=500)
            mock_ds2.select = Mock(return_value=selected2)
            
            mock_load.side_effect = [{'train': mock_ds1}, {'train': mock_ds2}]
            
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=1000)
            mock_concat.return_value = mock_combined
            
            # Test with single int (applies to all)
            dataset = GraphDataset(
                dataset_path=["data/test1.parquet", "data/test2.parquet"],
                max_samples_per_dataset=500
            )
            
            # Both should be limited to 500
            assert mock_ds1.select.call_count == 1
            assert mock_ds2.select.call_count == 1
            
            # Verify both get 500 samples
            call_args1 = mock_ds1.select.call_args[0][0]
            call_args2 = mock_ds2.select.call_args[0][0]
            assert len(call_args1) == 500
            assert len(call_args2) == 500


def test_graphdataset_no_limit():
    """Test that GraphDataset works without max_samples (uses all data)."""
    with patch('src.data_loader.aa_dataset.load_dataset') as mock_load:
        with patch('src.data_loader.aa_dataset.concatenate_datasets') as mock_concat:
            # Mock datasets
            mock_ds1 = Mock()
            mock_ds1.__len__ = Mock(return_value=1000)
            
            mock_ds2 = Mock()
            mock_ds2.__len__ = Mock(return_value=500)
            
            mock_load.side_effect = [{'train': mock_ds1}, {'train': mock_ds2}]
            
            mock_combined = Mock()
            mock_combined.__len__ = Mock(return_value=1500)
            mock_concat.return_value = mock_combined
            
            # Test without max_samples
            dataset = GraphDataset(
                dataset_path=["data/test1.parquet", "data/test2.parquet"],
                max_samples_per_dataset=None
            )
            
            # Should use full datasets
            assert mock_concat.call_count == 1


def test_modality_aware_sampler_with_concat_dataset():
    """Test that ModalityAwareBatchSampler works with ConcatDataset."""
    from torch.utils.data import ConcatDataset
    
    # Create mock datasets
    mock_ds1 = Mock()
    mock_ds1.__len__ = Mock(return_value=30)
    def getitem1(idx):
        return {'modality': 'protein', 'value': {}}
    mock_ds1.__getitem__ = Mock(side_effect=getitem1)
    
    mock_ds2 = Mock()
    mock_ds2.__len__ = Mock(return_value=20)
    def getitem2(idx):
        return {'modality': 'molecule', 'value': {}}
    mock_ds2.__getitem__ = Mock(side_effect=getitem2)
    
    # Create ConcatDataset
    concat_ds = ConcatDataset([mock_ds1, mock_ds2])
    
    # Create sampler
    sampler = ModalityAwareBatchSampler(
        dataset=concat_ds,
        batch_size=10,
        shuffle=False,
        seed=42,
    )
    
    # Should group by modality
    assert len(sampler.modality_indices) == 2
    assert 'protein' in sampler.modality_indices
    assert 'molecule' in sampler.modality_indices


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

