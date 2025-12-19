import pytest
from pathlib import Path

from src.data_factory.protein.map_fetch_pdb3d import blast_search_pdb


class TestBlastSearchPDB:
    """Tests for BLAST-based PDB search functionality."""
    
    def test_invalid_sequence_empty(self):
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="Invalid protein sequence"):
            blast_search_pdb("")
    
    def test_invalid_sequence_non_alpha(self):
        """Test that sequence with numbers raises ValueError."""
        with pytest.raises(ValueError, match="Invalid protein sequence"):
            blast_search_pdb("MKFL123KFS")
    
    def test_missing_biopython_import(self, monkeypatch):
        """Test graceful handling when Biopython is not installed."""
        import builtins
        real_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if 'Bio.Blast' in name or name == 'Bio.Blast':
                raise ImportError("No module named 'Bio'")
            return real_import(name, *args, **kwargs)
        
        monkeypatch.setattr(builtins, '__import__', mock_import)
        
        with pytest.raises(ImportError, match="Biopython required"):
            # Force reimport to trigger the mock
            import importlib
            import src.data_factory.protein.map_fetch_pdb3d as map_fetch_pdb3d
            importlib.reload(map_fetch_pdb3d)
            map_fetch_pdb3d.blast_search_pdb("MKFLKFS")
    
    # @pytest.mark.integration
    # @pytest.mark.slow
    # def test_blast_search_known_sequence(self):
    #     """
    #     Integration test with known protein sequence.
    #     Uses a short, well-characterized sequence (insulin signal peptide).
        
    #     This test makes real API calls and may be slow (~10-30 seconds).
    #     Skip by default: pytest -m "not integration"
    #     """
    #     # Human insulin signal peptide (24 aa)
    #     sequence = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKA"
        
    #     results = blast_search_pdb(
    #         sequence,
    #         e_value_threshold=1.0,  # Relaxed for short sequence
    #         max_hits=5
    #     )
        
    #     # Assertions
    #     assert isinstance(results, list), "Results should be a list"
    #     assert len(results) <= 5, "Should return at most max_hits results"
        
    #     if len(results) > 0:
    #         # Check result structure
    #         for pdb_id, e_value, identity in results:
    #             assert isinstance(pdb_id, str), "PDB ID should be string"
    #             assert len(pdb_id) == 4, f"PDB ID should be 4 characters, got {pdb_id}"
    #             assert pdb_id.isupper(), "PDB ID should be uppercase"
                
    #             assert isinstance(e_value, float), "E-value should be float"
    #             assert e_value >= 0, "E-value should be non-negative"
                
    #             assert isinstance(identity, float), "Identity should be float"
    #             assert 0 <= identity <= 100, f"Identity should be 0-100%, got {identity}"
            
    #         # Check sorting (best e-values first)
    #         e_values = [e for _, e, _ in results]
    #         assert e_values == sorted(e_values), "Results should be sorted by e-value"
            
    #         print(f"\nFound {len(results)} matches:")
    #         for pdb_id, e_val, identity in results[:3]:
    #             print(f"  {pdb_id}: E={e_val:.2e}, Identity={identity:.1f}%")
    
    # @pytest.mark.integration
    # @pytest.mark.slow
    # def test_blast_search_ubiquitin(self):
    #     """
    #     Integration test with ubiquitin sequence (highly conserved, many PDB entries).
    #     """
    #     # Human ubiquitin (76 aa) - should have many hits
    #     ubiquitin = (
    #         "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    #     )
        
    #     results = blast_search_pdb(
    #         ubiquitin,
    #         e_value_threshold=0.001,
    #         max_hits=10
    #     )
        
    #     assert len(results) > 0, "Ubiquitin should have matches in PDB"
    #     assert len(results) <= 10, "Should respect max_hits"
        
    #     # For ubiquitin, expect high identity matches
    #     top_match = results[0]
    #     pdb_id, e_value, identity = top_match
        
    #     assert e_value < 1e-10, f"Top match should have very low e-value, got {e_value}"
    #     assert identity > 80, f"Top match should have >80% identity, got {identity}"
        
    #     print(f"\nUbiquitin top match: {pdb_id} (E={e_value:.2e}, {identity:.1f}%)")
    
    # @pytest.mark.integration  
    # @pytest.mark.slow
    # def test_blast_search_no_hits(self):
    #     """Test with artificial sequence unlikely to have hits."""
    #     # Random, non-natural sequence
    #     fake_sequence = "AAAAAKKKKKEEEEEGGGGG"
        
    #     results = blast_search_pdb(
    #         fake_sequence,
    #         e_value_threshold=0.001,
    #         max_hits=5
    #     )
        
    #     # May or may not have hits, but should not crash
    #     assert isinstance(results, list)
    #     assert len(results) <= 5
    
    # def test_valid_sequence_format(self):
    #     """Test that valid amino acid sequences are accepted."""
    #     # These should not raise exceptions (but we won't run actual BLAST)
    #     valid_sequences = [
    #         "MKFLKFSLLTAVLLSVVFAFSSCG",
    #         "mkflkfslltavllsvvfafsscg",  # lowercase
    #         "MKF LKF SLL",  # with spaces (should be valid after strip)
    #     ]
        
    #     # We just check the validation part doesn't raise
    #     for seq in valid_sequences:
    #         if seq.replace(' ', '').isalpha():
    #             assert True, f"Sequence {seq} should be valid format"


if __name__ == '__main__':
    # Run integration tests manually
    print("Running integration test with real BLAST query...")
    print("This may take 10-30 seconds...\n")
    
    test = TestBlastSearchPDB()
    test.test_blast_search_known_sequence()
