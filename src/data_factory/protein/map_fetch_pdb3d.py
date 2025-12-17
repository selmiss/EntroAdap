import json
import os
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError


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


def download_pdb_structures(pdb_ids: list, output_dir: str, file_format: str = 'cif'):
    """
    Download mmCIF 3D structure files for given PDB IDs.
    
    Args:
        pdb_ids: List of PDB IDs to download
        output_dir: Directory to save the downloaded files
        file_format: Format to download ('cif' for mmCIF or 'pdb' for PDB format)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://files.rcsb.org/download"
    extension = 'cif' if file_format == 'cif' else 'pdb'
    
    success_count = 0
    failed_ids = []
    
    for pdb_id in pdb_ids:
        try:
            url = f"{base_url}/{pdb_id}.{extension}"
            output_file = os.path.join(output_dir, f"{pdb_id}.{extension}")
            
            if os.path.exists(output_file):
                print(f"Skipping {pdb_id} - already exists")
                continue
            
            urlretrieve(url, output_file)
            success_count += 1
            print(f"Downloaded {pdb_id}.{extension}")
            
        except HTTPError as e:
            print(f"Failed to download {pdb_id}: HTTP {e.code}")
            failed_ids.append(pdb_id)
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
            failed_ids.append(pdb_id)
    
    print(f"\nDownload complete: {success_count}/{len(pdb_ids)} succeeded")
    if failed_ids:
        print(f"Failed IDs: {', '.join(failed_ids)}")


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
