import json
import os
import time
import gzip
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import HTTPError, URLError
from typing import List, Tuple, Optional
from io import StringIO


def blast_search_pdb(protein_sequence: str, 
                      e_value_threshold: float = 0.001,
                      max_hits: int = 10) -> List[Tuple[str, float, float]]:
    """
    Search for similar protein sequences in PDB database using BLAST.
    
    Args:
        protein_sequence: Amino acid sequence (single letter code)
        e_value_threshold: E-value threshold for filtering results (default: 0.001)
        max_hits: Maximum number of hits to return (default: 10)
        
    Returns:
        List of tuples: [(pdb_id, e_value, identity_percentage), ...]
        Sorted by e-value (best matches first)
        
    Example:
        >>> results = blast_search_pdb("MKFLKFSLLTAVLLSVVFAFSSCGDDDDTYPYDVPDYAIE")
        >>> for pdb_id, e_val, identity in results[:5]:
        ...     print(f"{pdb_id}: E={e_val:.2e}, Identity={identity:.1f}%")
    """
    try:
        from Bio.Blast import NCBIWWW, NCBIXML
    except ImportError:
        raise ImportError("Biopython required. Install: pip install biopython")
    
    if not protein_sequence or not protein_sequence.replace(' ', '').isalpha():
        raise ValueError("Invalid protein sequence")
    
    print(f"Running BLAST search (length={len(protein_sequence)})...")
    
    # Run BLAST against PDB database
    result_handle = NCBIWWW.qblast(
        program="blastp",
        database="pdb",
        sequence=protein_sequence,
        expect=e_value_threshold,
        hitlist_size=max_hits
    )
    
    # Parse results
    blast_records = NCBIXML.parse(result_handle)
    results = []
    
    for blast_record in blast_records:
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                if hsp.expect <= e_value_threshold:
                    # Extract PDB ID from title (format: "pdb|1ABC|A")
                    title = alignment.title
                    pdb_id = None
                    if '|' in title:
                        parts = title.split('|')
                        if len(parts) >= 2:
                            pdb_id = parts[1][:4].upper()  # First 4 chars
                    
                    if pdb_id:
                        identity_pct = (hsp.identities / hsp.align_length) * 100
                        results.append((pdb_id, hsp.expect, identity_pct))
    
    # Remove duplicates and sort by e-value
    unique_results = {}
    for pdb_id, e_val, identity in results:
        if pdb_id not in unique_results or e_val < unique_results[pdb_id][0]:
            unique_results[pdb_id] = (e_val, identity)
    
    final_results = [(pdb_id, e_val, identity) 
                     for pdb_id, (e_val, identity) in unique_results.items()]
    final_results.sort(key=lambda x: x[1])  # Sort by e-value
    
    print(f"Found {len(final_results)} unique PDB matches")
    return final_results[:max_hits]


def extract_pdb_ids(mapping_file_path: str):
    """
    Extract PDB IDs from UniProt to PDB mapping file.
    
    Args:
        mapping_file_path: Path to the ID mapping JSON file
        
    Returns:
        dict: Dictionary mapping UniProt IDs to list of PDB IDs
        list: Unique list of all PDB IDs
    """
    with open(mapping_file_path, 'r') as f:
        data = json.load(f)
    
    mapping = {}
    all_pdb_ids = set()
    
    for entry in data.get('results', []):
        try:
            uniprot_id = entry.get('from')
            pdb_id = entry.get('to')
            
            if not uniprot_id or not pdb_id:
                continue
            
            if uniprot_id not in mapping:
                mapping[uniprot_id] = []
            
            mapping[uniprot_id].append(pdb_id)
            all_pdb_ids.add(pdb_id)
            
        except Exception as e:
            print(f"Error processing entry {entry}: {e}")
            continue
    
    return mapping, sorted(list(all_pdb_ids))


def _try_batch_download(ids: list, out_dir: Path, fmt: str, verbose: bool) -> bool:
    """Try official batch download script, decompress, and cleanup. Returns True if successful."""
    # Download script if missing
    script_path = out_dir / "batch_download.sh"
    script_url = "https://www.rcsb.org/scripts/batch_download.sh"
    if not script_path.exists():
        urlretrieve(script_url, script_path)
        script_path.chmod(script_path.stat().st_mode | 0o111)

    # Write ID list (comma-separated, lowercase for batch script)
    list_file = out_dir / "pdb_ids.txt"
    list_file.write_text(",".join([x.lower() for x in ids]))

    # Run batch script
    cmd = ["bash", str(script_path), "-f", str(list_file)]
    cmd += ["-c"] if fmt == "cif" else ["-p"]

    try:
        result = subprocess.run(cmd, cwd=str(out_dir), check=True, capture_output=True, text=True)
        if verbose:
            print("Batch download completed, decompressing...")
        
        # Decompress .gz files
        _decompress_batch_files(ids, out_dir, fmt, verbose)
        return True
        
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Batch script failed: {e.stderr if e.stderr else e}")
        return False
    except Exception as e:
        if verbose:
            print(f"Batch download error: {e}")
        return False


def _decompress_batch_files(ids: list, out_dir: Path, fmt: str, verbose: bool):
    """Decompress .gz files from batch download and cleanup."""
    success = 0
    skipped = 0
    failed = []

    for pdb_id in ids:
        out_file = out_dir / f"{pdb_id}.{fmt}"
        
        # Skip if already exists
        if out_file.exists():
            skipped += 1
            continue

        # Find .gz file (various naming conventions)
        gz_candidates = [
            out_dir / f"{pdb_id.lower()}.{fmt}.gz",
            out_dir / f"{pdb_id}.{fmt}.gz",
            out_dir / f"pdb{pdb_id.lower()}.ent.gz" if fmt == "pdb" else out_dir / f"pdb_0000{pdb_id.lower()}.cif.gz",
        ]
        
        gz_path = next((p for p in gz_candidates if p.exists()), None)
        if gz_path is None:
            failed.append(pdb_id)
            continue

        try:
            with gzip.open(gz_path, "rb") as fin:
                with open(out_file, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            
            # Cleanup .gz file
            gz_path.unlink()
            success += 1
            
        except Exception as e:
            failed.append(pdb_id)
            if out_file.exists():
                out_file.unlink()

    if verbose:
        print(f"Decompressed: {success} new, {skipped} existed, {len(failed)} failed")


def _individual_downloads(ids: list, out_dir: Path, fmt: str, delay: float, verbose: bool):
    """Download files individually with rate limiting."""
    base_url = "https://files.rcsb.org/download"
    ext = fmt
    user_agent = 'Mozilla/5.0 (compatible; PDB-Downloader/1.0)'
    
    success = 0
    skipped = 0
    failed = []

    for idx, pdb_id in enumerate(ids, 1):
        output_file = out_dir / f"{pdb_id}.{ext}"
        
        # Skip if exists
        if output_file.exists():
            skipped += 1
            if verbose and (idx % 100 == 0 or idx <= 5):
                print(f"[{idx}/{len(ids)}] Skipping {pdb_id} - exists")
            continue
        
        # Download
        url = f"{base_url}/{pdb_id}.{ext}"
        try:
            req = Request(url, headers={'User-Agent': user_agent})
            with urlopen(req, timeout=30) as response:
                with open(output_file, 'wb') as f:
                    f.write(response.read())
            
            success += 1
            if verbose and (idx % 100 == 0 or idx <= 5):
                print(f"[{idx}/{len(ids)}] Downloaded {pdb_id}.{ext}")
            
            # Rate limiting
            if idx < len(ids):
                time.sleep(delay)
                
        except Exception as e:
            failed.append(pdb_id)
            if verbose:
                print(f"[{idx}/{len(ids)}] Failed {pdb_id}: {e}")
            if output_file.exists():
                output_file.unlink()

    if verbose:
        print(f"\nDownload complete: {success} new, {skipped} skipped, {len(failed)} failed")
        if failed and len(failed) <= 20:
            print(f"Failed IDs: {', '.join(failed)}")


def download_pdb_structures(
    pdb_ids: list,
    output_dir: str,
    file_format: str = "cif",
    delay: float = 0.1,
    verbose: bool = True,
):
    """
    Download PDB structures using official batch script or fallback to individual downloads.

    Tries official RCSB batch_download.sh first (efficient for large batches), falls back 
    to individual downloads if batch method fails (e.g., missing curl/wget).

    Args:
      - pdb_ids: List of PDB IDs to download
      - output_dir: Directory to save files
      - file_format: 'cif' or 'pdb'
      - delay: Delay between requests in seconds (for individual downloads)
      - verbose: Print progress messages
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fmt = file_format.lower()
    if fmt not in {"cif", "pdb"}:
        raise ValueError("file_format must be 'cif' or 'pdb'")

    # Normalize IDs
    ids = [str(x).strip().upper() for x in pdb_ids if str(x).strip()]
    if not ids:
        if verbose:
            print("No PDB IDs provided.")
        return

    # Try batch download first (efficient for large batches)
    batch_success = False
    if len(ids) > 10:  # Only use batch for >10 files
        try:
            if verbose:
                print(f"Attempting batch download for {len(ids)} structures...")
            batch_success = _try_batch_download(ids, out, fmt, verbose)
        except Exception as e:
            if verbose:
                print(f"Batch download failed: {e}")
                print("Falling back to individual downloads...")
    
    # If batch failed or not attempted, use individual downloads
    if not batch_success:
        if verbose and len(ids) <= 10:
            print(f"Using individual downloads for {len(ids)} structures...")
        _individual_downloads(ids, out, fmt, delay, verbose)


def download_pdb_structures_old(pdb_ids: list, output_dir: str, file_format: str = 'cif', 
                           delay: float = 0.1, verbose: bool = True):
    """
    Download mmCIF 3D structure files for given PDB IDs with rate limiting.
    
    Args:
        pdb_ids: List of PDB IDs to download
        output_dir: Directory to save the downloaded files
        file_format: Format to download ('cif' for mmCIF or 'pdb' for PDB format)
        delay: Delay between requests in seconds (default: 0.1, recommended: 0.1-0.5)
        verbose: Print progress messages (default: True)
        
    Note:
        RCSB PDB is a free public resource. Please be respectful:
        - Use delay >= 0.1 seconds between requests (~10 req/sec max)
        - For very large downloads (>10k files), consider using rsync instead
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://files.rcsb.org/download"
    extension = 'cif' if file_format == 'cif' else 'pdb'
    
    # User-Agent header for politeness
    user_agent = 'Mozilla/5.0 (compatible; PDB-Downloader/1.0)'
    
    success_count = 0
    failed_ids = []
    
    for idx, pdb_id in enumerate(pdb_ids, 1):
        try:
            output_file = os.path.join(output_dir, f"{pdb_id}.{extension}")
            
            if os.path.exists(output_file):
                if verbose and (idx % 100 == 0 or idx <= 10):
                    print(f"Skipping {pdb_id} - already exists ({idx}/{len(pdb_ids)})")
                continue
            
            url = f"{base_url}/{pdb_id}.{extension}"
            
            # Create request with User-Agent header
            req = Request(url, headers={'User-Agent': user_agent})
            
            # Download file
            with urlopen(req, timeout=30) as response:
                with open(output_file, 'wb') as f:
                    f.write(response.read())
            
            success_count += 1
            if verbose and (idx % 100 == 0 or idx <= 10):
                print(f"Downloaded {pdb_id}.{extension} ({idx}/{len(pdb_ids)})")
            
            # Rate limiting - be polite to the server
            if idx < len(pdb_ids):  # Don't sleep after last download
                time.sleep(delay)
            
        except HTTPError as e:
            if verbose:
                print(f"Failed to download {pdb_id}: HTTP {e.code}")
            failed_ids.append(pdb_id)
            # Clean up failed download
            if os.path.exists(output_file):
                os.remove(output_file)
        except Exception as e:
            if verbose:
                print(f"Error downloading {pdb_id}: {e}")
            failed_ids.append(pdb_id)
            # Clean up failed download
            if os.path.exists(output_file):
                os.remove(output_file)
    
    if verbose:
        print(f"\nDownload complete: {success_count}/{len(pdb_ids)} succeeded")
        if failed_ids:
            print(f"Failed IDs ({len(failed_ids)}): {', '.join(failed_ids[:20])}")
            if len(failed_ids) > 20:
                print(f"  ... and {len(failed_ids) - 20} more")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python map_fetch_pdb3d.py <mapping_json_path>")
        sys.exit(1)
    
    mapping_file = sys.argv[1]
    
    print("Extracting PDB IDs...")
    mapping, pdb_ids = extract_pdb_ids(mapping_file)
    
    print(f"\nFound {len(mapping)} UniProt IDs mapped to {len(pdb_ids)} unique PDB IDs")
    print(f"PDB IDs: {', '.join(pdb_ids[:10])}{'...' if len(pdb_ids) > 10 else ''}")
    
    mapping_dir = Path(mapping_file).parent
    output_dir = mapping_dir / 'pdb_structures'
    
    print(f"\nDownloading structures to {output_dir}...")
    download_pdb_structures(pdb_ids, str(output_dir))
