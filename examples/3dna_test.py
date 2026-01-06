import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def _clean_dna(seq: str) -> str:
    """Validate and clean DNA sequence (A/C/G/T only)."""
    s = re.sub(r"\s+", "", seq).upper()
    if not s:
        raise ValueError("Empty sequence.")
    bad = sorted(set(ch for ch in s if ch not in {"A", "C", "G", "T"}))
    if bad:
        raise ValueError(f"fiber expects DNA letters A/C/G/T. Found: {bad}")
    return s


def _clean_rna(seq: str) -> str:
    """Validate and clean RNA sequence (A/C/G/U only)."""
    s = re.sub(r"\s+", "", seq).upper()
    if not s:
        raise ValueError("Empty sequence.")
    bad = sorted(set(ch for ch in s if ch not in {"A", "C", "G", "U"}))
    if bad:
        raise ValueError(f"fiber expects RNA letters A/C/G/U. Found: {bad}")
    return s


def test_3dna_fiber(fiber_exe: str = "fiber") -> Dict[str, str]:
    """
    Sanity check that `fiber` is installed and runnable.
    Returns a dict with path and a short help/version snippet.
    """
    path = shutil.which(fiber_exe)
    if path is None:
        raise RuntimeError(f"Cannot find `{fiber_exe}` in PATH. Check your X3DNA PATH export.")
    try:
        # Note: fiber -h returns exit status 1, but still prints help successfully
        p = subprocess.run(
            [fiber_exe, "-h"],
            check=False,  # Don't check exit code, some tools return non-zero for help
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Cannot find `{fiber_exe}` in PATH.") from e

    out = (p.stdout or "").strip()
    # Verify we got actual help output
    if not out or "fiber" not in out.lower():
        raise RuntimeError(f"`{fiber_exe} -h` produced no valid output.")
    
    snippet = "\n".join(out.splitlines()[:25])
    return {"fiber_path": path, "fiber_help_head": snippet}


def dna_to_pdb(seq: str, out_pdb: str, fiber_exe: str = "fiber") -> str:
    """
    Uses X3DNA `fiber` to build a canonical B-DNA model from sequence and write PDB.
    """
    seq = _clean_dna(seq)
    out_pdb = str(Path(out_pdb).resolve())
    Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [fiber_exe, f"-seq={seq}", out_pdb],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Cannot find `{fiber_exe}` in PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"`fiber` failed.\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}") from e

    if not os.path.exists(out_pdb) or os.path.getsize(out_pdb) == 0:
        raise RuntimeError("fiber produced empty output PDB.")
    return out_pdb


def rna_to_pdb(seq: str, out_pdb: str, fiber_exe: str = "fiber", single_strand: bool = True) -> str:
    """
    Uses X3DNA `fiber` to build a canonical A-RNA model from sequence and write PDB.
    
    Args:
        seq: RNA sequence (A/C/G/U)
        out_pdb: Output PDB file path
        fiber_exe: Path to fiber executable
        single_strand: Generate single-stranded RNA (default: True, biologically correct)
    
    Returns:
        Path to generated PDB file
    """
    seq = _clean_rna(seq)
    out_pdb = str(Path(out_pdb).resolve())
    out_dir = Path(out_pdb).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # fiber writes temp files to cwd, so we need to run it from the output directory
    old_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        
        # Use -rna flag for A-form RNA structure
        # Use -single flag to generate single-stranded RNA (biological default)
        cmd = [fiber_exe, "-rna", f"-seq={seq}"]
        if single_strand:
            cmd.append("-single")
        cmd.append(Path(out_pdb).name)  # Use relative path since we're in the output dir
        
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"Cannot find `{fiber_exe}` in PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"`fiber` failed.\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}") from e
    finally:
        os.chdir(old_cwd)

    if not os.path.exists(out_pdb) or os.path.getsize(out_pdb) == 0:
        raise RuntimeError("fiber produced empty output PDB.")
    return out_pdb


def pdb_to_atom_coords(pdb_path: str) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """
    Parses ATOM/HETATM records from PDB.
    Returns:
      coords: (N,3) float32
      meta: list of dicts per atom with atom_name, res_name, chain_id, res_id, element
    """
    coords = []
    meta = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = line[0:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue

            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = (line[21:22].strip() or "?")
            res_id = line[22:26].strip()

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            element = (line[76:78].strip() if len(line) >= 78 else "")
            coords.append((x, y, z))
            meta.append(
                {
                    "atom_name": atom_name,
                    "res_name": res_name,
                    "chain_id": chain_id,
                    "res_id": res_id,
                    "element": element,
                }
            )

    if not coords:
        raise RuntimeError(f"No ATOM/HETATM records parsed from {pdb_path}")
    return np.asarray(coords, dtype=np.float32), meta


def pdb_to_mmcif(pdb_path: str, out_cif: str) -> str:
    """
    Converts PDB to mmCIF via gemmi, if available.
    If gemmi is not installed, raises with an install hint.
    """
    try:
        import gemmi  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "mmCIF writing requires `gemmi`.\n"
            "Install: pip install gemmi\n"
            "Then re-run."
        ) from e

    out_cif = str(Path(out_cif).resolve())
    Path(out_cif).parent.mkdir(parents=True, exist_ok=True)

    st = gemmi.read_structure(pdb_path)
    st.make_mmcif_document().write_file(out_cif)
    if not os.path.exists(out_cif) or os.path.getsize(out_cif) == 0:
        raise RuntimeError("mmCIF conversion produced empty file.")
    return out_cif


def rna_sequence_to_coords_or_cif(
    seq: str,
    workdir: str = "./x3dna_test_rna",
    write_cif: bool = False,
    fiber_exe: str = "fiber",
    single_strand: bool = True,
) -> Dict[str, object]:
    """
    End-to-end RNA processing:
      RNA sequence -> fiber PDB (A-form) -> coords (+ optional CIF)

    Args:
        seq: RNA sequence (A/C/G/U)
        workdir: Working directory for output files
        write_cif: Whether to also generate mmCIF format
        fiber_exe: Path to fiber executable
        single_strand: Generate single-stranded RNA (default: True, biologically correct)
        
    Returns:
        dict with: fiber_info, pdb_path, coords (N,3), meta (list), and optionally cif_path.
    """
    Path(workdir).mkdir(parents=True, exist_ok=True)

    # 1) verify fiber works
    info = test_3dna_fiber(fiber_exe=fiber_exe)

    # 2) build RNA PDB
    pdb_path = rna_to_pdb(seq, out_pdb=str(Path(workdir) / "model_rna.pdb"), fiber_exe=fiber_exe, single_strand=single_strand)

    # 3) parse coords
    coords, meta = pdb_to_atom_coords(pdb_path)

    out = {"fiber_info": info, "pdb_path": pdb_path, "coords": coords, "meta": meta}

    # 4) optional CIF
    if write_cif:
        cif_path = pdb_to_mmcif(pdb_path, out_cif=str(Path(workdir) / "model_rna.cif"))
        out["cif_path"] = cif_path

    return out


def dna_sequence_to_coords_or_cif(
    seq: str,
    workdir: str = "./x3dna_test",
    write_cif: bool = False,
    fiber_exe: str = "fiber",
) -> Dict[str, object]:
    """
    End-to-end DNA processing:
      DNA sequence -> fiber PDB (B-form) -> coords (+ optional CIF)

    Args:
        seq: DNA sequence (A/C/G/T)
        workdir: Working directory for output files
        write_cif: Whether to also generate mmCIF format
        fiber_exe: Path to fiber executable
        
    Returns:
        dict with: fiber_info, pdb_path, coords (N,3), meta (list), and optionally cif_path.
    """
    Path(workdir).mkdir(parents=True, exist_ok=True)

    # 1) verify fiber works
    info = test_3dna_fiber(fiber_exe=fiber_exe)

    # 2) build PDB
    pdb_path = dna_to_pdb(seq, out_pdb=str(Path(workdir) / "model.pdb"), fiber_exe=fiber_exe)

    # 3) parse coords
    coords, meta = pdb_to_atom_coords(pdb_path)

    out = {"fiber_info": info, "pdb_path": pdb_path, "coords": coords, "meta": meta}

    # 4) optional CIF
    if write_cif:
        cif_path = pdb_to_mmcif(pdb_path, out_cif=str(Path(workdir) / "model.cif"))
        out["cif_path"] = cif_path

    return out


def nucleic_acid_to_coords_or_cif(
    seq: str,
    seq_type: str = "auto",
    workdir: str = "./x3dna_test_na",
    write_cif: bool = False,
    fiber_exe: str = "fiber",
    rna_single_strand: bool = True,
) -> Dict[str, object]:
    """
    Generic nucleic acid processing that handles both DNA and RNA.
    
    Args:
        seq: Nucleic acid sequence (A/C/G/T for DNA or A/C/G/U for RNA)
        seq_type: "dna", "rna", or "auto" (auto-detects based on T vs U)
        workdir: Working directory for output files
        write_cif: Whether to also generate mmCIF format
        fiber_exe: Path to fiber executable
        rna_single_strand: Generate RNA as single-stranded (default: True, biologically correct)
        
    Returns:
        dict with: seq_type, fiber_info, pdb_path, coords (N,3), meta (list), and optionally cif_path.
    """
    seq_clean = re.sub(r"\s+", "", seq).upper()
    
    # Auto-detect sequence type
    if seq_type == "auto":
        has_t = "T" in seq_clean
        has_u = "U" in seq_clean
        
        if has_u and not has_t:
            seq_type = "rna"
        elif has_t and not has_u:
            seq_type = "dna"
        elif has_u and has_t:
            raise ValueError("Sequence contains both T and U - cannot auto-detect type. Please specify seq_type explicitly.")
        else:
            # No T or U - could be either, default to DNA
            seq_type = "dna"
    
    seq_type = seq_type.lower()
    
    if seq_type == "dna":
        result = dna_sequence_to_coords_or_cif(seq, workdir, write_cif, fiber_exe)
    elif seq_type == "rna":
        result = rna_sequence_to_coords_or_cif(seq, workdir, write_cif, fiber_exe, single_strand=rna_single_strand)
    else:
        raise ValueError(f"seq_type must be 'dna', 'rna', or 'auto'. Got: {seq_type}")
    
    result["seq_type"] = seq_type
    return result


if __name__ == "__main__":
    # Allow specifying fiber path via environment variable or use default
    fiber_exe = os.environ.get("X3DNA_FIBER", "fiber")
    
    print("="*60)
    print("Testing DNA sequence processing")
    print("="*60)
    dna_seq = "ATTCAGATTGCCTCTCATTGTCTCACCCATATTATGGGAACCAAATATGAGC"
    result_dna = dna_sequence_to_coords_or_cif(dna_seq, write_cif=False, fiber_exe=fiber_exe)

    print("fiber:", result_dna["fiber_info"]["fiber_path"])
    print("pdb:", result_dna["pdb_path"])
    coords_dna = result_dna["coords"]
    print("coords shape:", coords_dna.shape, coords_dna.dtype)
    print("first atom meta:", result_dna["meta"][0])
    print("first atom xyz:", coords_dna[0].tolist())
    
    print("\n" + "="*60)
    print("Testing RNA sequence processing")
    print("="*60)
    # Convert DNA sequence to RNA (T -> U) for testing
    rna_seq = "AUUCAGAUUGCCUCUCAUUGUCUCACCCAUAUUAUGGGAACCAAAUAUGAGC"
    result_rna = rna_sequence_to_coords_or_cif(rna_seq, write_cif=False, fiber_exe=fiber_exe)
    
    print("fiber:", result_rna["fiber_info"]["fiber_path"])
    print("pdb:", result_rna["pdb_path"])
    coords_rna = result_rna["coords"]
    print("coords shape:", coords_rna.shape, coords_rna.dtype)
    print("first atom meta:", result_rna["meta"][0])
    print("first atom xyz:", coords_rna[0].tolist())
    
    print("\n" + "="*60)
    print("Testing auto-detection (should detect as RNA)")
    print("="*60)
    result_auto = nucleic_acid_to_coords_or_cif(rna_seq, seq_type="auto", fiber_exe=fiber_exe)
    print("Detected type:", result_auto["seq_type"])
    print("pdb:", result_auto["pdb_path"])
