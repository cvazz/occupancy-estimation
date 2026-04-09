#!/usr/bin/env python3
import sys
import os
import shutil
import subprocess
import re
import logging
import argparse
from pathlib import Path

# Configure logger to print clean messages to the console
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, log_name, cwd):
    """Executes a shell command inside a specific working directory."""
    log_path = cwd / log_name
    
    # Convert all command list items to strings FIRST!
    cmd_str = [str(c) for c in cmd]
    
    # Now it's safe to join them for the logger
    logger.info(f"Running: {' '.join(cmd_str[:2])} ... (Log: {log_path})")
    
    with open(log_path, "w") as log:
        subprocess.run(cmd_str, stdout=log, stderr=sys.stdout, cwd=cwd)

def extract_statistics(refine_log, val_log):
    """Parses standard Phenix log files to extract the required metrics."""
    stats = {k: "N/A" for k in ["rwork", "rfree", "dpi", "bonds", "angles", "local_cc", "clash", "rama", "rota"]}
    
    if refine_log.exists():
        with open(refine_log, 'r') as f:
            content = f.read()
            
        # 1. R-work and R-free
        r_matches = re.findall(r"Final R-work =\s*([\d.]+),\s*R-free =\s*([\d.]+)", content)
        if r_matches: 
            stats["rwork"], stats["rfree"] = r_matches[-1]
            
        # 2. Cruickshank DPI
        dpi_matches = re.findall(r"coordinate error.*?:\s*([\d.]+)", content)
        if dpi_matches: 
            stats["dpi"] = dpi_matches[-1]
            
        # 3. Bonds and Angles RMSD
        end_matches = re.findall(r"^\s*end:\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", content, re.MULTILINE)
        if end_matches: 
            stats["bonds"], stats["angles"] = end_matches[-1]
            
        # 4. Clashscore, Ramachandran, Rotamers (from the XYZ individual table)
        # Matches the columns: work, free, delta, bonds, angl, CLASH, RAMA, ROTA
        xyz_matches = re.findall(r"^\s*\d+\.\d+\s+\d+\.\d+\s+-?\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", content, re.MULTILINE)
        if xyz_matches:
            stats["clash"], stats["rama"], stats["rota"] = xyz_matches[-1]

    # Map CC is the only thing we still rel on validation logs or PDB headers for
    if val_log.exists():
        with open(val_log, 'r') as f:
            content = f.read()
            
        # Catch various Map CC formatting
        cc_match = re.search(r"local CC[\s:=]*([\d.]+)", content, re.IGNORECASE)
        if cc_match: 
            stats["local_cc"] = cc_match.group(1)


        
    return stats

def print_markdown_table(vac_stats, x8_stats):
    """Formats the extracted statistics into a Markdown table."""
    if not vac_stats or not x8_stats:
        return
    print(f"""
| Metric | Vacuum | x8 |
| :--- | :--- | :--- |
| **R-work** | {vac_stats['rwork']} | {x8_stats['rwork']} |
| **R-free** | {vac_stats['rfree']} | {x8_stats['rfree']} |
| **Map CC Local** | {vac_stats['local_cc']} | {x8_stats['local_cc']} |
| **Cruickshank DPI (Å)** | {vac_stats['dpi']} | {x8_stats['dpi']} |
| **Bonds RMSD (Å)** | {vac_stats['bonds']} | {x8_stats['bonds']} |
| **Angles RMSD (°)** | {vac_stats['angles']} | {x8_stats['angles']} |
| **Clashscore** | {vac_stats['clash']} | {x8_stats['clash']} |
| **Ramachandran Outliers (%)** | {vac_stats['rama']} | {x8_stats['rama']} |
| **Rotamer Outliers (%)** | {vac_stats['rota']} | {x8_stats['rota']} |
""")


def run_comparison(pdb_file, mtz_vacuum, mtz_x8, flags_file=None, output_dir="."):
    """
    Core API Function with step-by-step checkpointing.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pdb_abs = Path(pdb_file).resolve()
    mtz_vac_abs = Path(mtz_vacuum).resolve()
    mtz_x8_abs = Path(mtz_x8).resolve()
    
    if not all(p.exists() for p in [pdb_abs, mtz_vac_abs, mtz_x8_abs]):
        logger.error("Error: One or more input files do not exist.")
        return None, None

    working_pdb = out_dir / "working_copy.pdb"
    min_pdb = out_dir / "minimized.pdb"

    # Make a safe copy in the isolated directory
    shutil.copy(pdb_abs, working_pdb)

    logger.info(f"\n=== Processing in Sandbox: {out_dir.name} ===")

    # --- Step 1: Minimization Checkpoint ---
    if not min_pdb.exists():
        run_command(["phenix.geometry_minimization", working_pdb], "min.log", cwd=out_dir)
        generated_min = out_dir / "working_copy_minimized.pdb"
        if generated_min.exists():
            generated_min.rename(min_pdb)
        else:
            logger.error("Geometry minimization failed.")
            return None, None
    else:
        logger.info("-> Skipping Minimization (minimized.pdb already exists)")

    # --- Flags Setup ---
    if flags_file and str(flags_file).upper() != "GENERATE":
        flags_abs = str(Path(flags_file).resolve())
    elif (out_dir / "vacuum_flags.mtz").exists():
        flags_abs = str(out_dir / "vacuum_flags.mtz")
    else:
        flags_abs = "GENERATE"

    refine_cmd = [
        "phenix.refine", min_pdb, 
        "strategy=individual_sites+individual_adp",
        "main.number_of_macro_cycles=3",
        "output.serial=1", "--overwrite"
    ]

    # --- Step 2: Vacuum Refinement Checkpoint ---
    if not (out_dir / "refine_vacuum_001.pdb").exists():
        vac_cmd = refine_cmd + [mtz_vac_abs, "output.prefix=refine_vacuum"]
        if flags_abs == "GENERATE":
            vac_cmd.append("xray_data.r_free_flags.generate=True")
        else:
            vac_cmd.append(f"xray_data.r_free_flags.file_name={flags_abs}")
            
        run_command(vac_cmd, "refine_vacuum.log", cwd=out_dir)
    else:
        logger.info("-> Skipping Vacuum Refinement (refine_vacuum_001.pdb already exists)")

    # Update flags if we generated them
    if flags_abs == "GENERATE":
        flags_abs = str(out_dir / "refine_vacuum_001.mtz")

    # --- Step 3: x8 Refinement Checkpoint ---
    if not (out_dir / "refine_x8_001.pdb").exists():
        x8_cmd = refine_cmd + [mtz_x8_abs, f"xray_data.r_free_flags.file_name={flags_abs}", "output.prefix=refine_x8"]
        run_command(x8_cmd, "refine_x8.log", cwd=out_dir)
    else:
        logger.info("-> Skipping x8 Refinement (refine_x8_001.pdb already exists)")

    # --- Step 4: Validation Checkpoint (with comprehensive=true) ---
    val_vac_log = out_dir / "val_vacuum.log"
    val_x8_log = out_dir / "val_x8.log"
    if not val_vac_log.exists() or val_vac_log.stat().st_size == 0:
    # if True:
        cmd_list = ["phenix.get_cc_mtz_pdb", "refine_vacuum_001.pdb", "refine_vacuum_001.mtz",]
        run_command(cmd_list, "val_vacuum.log", cwd=out_dir)
        print(f"Validation results for vacuum refinement (running {cmd_list})saved to {val_vac_log}")
    # else:
    #     logger.info("-> Skipping Vacuum Validation (val_vacuum.log already exists)")

    if not val_x8_log.exists() or val_x8_log.stat().st_size == 0:
        run_command(["phenix.get_cc_mtz_pdb", "refine_x8_001.pdb", "refine_x8_001.mtz",], "val_x8.log", cwd=out_dir)
    else:
        logger.info("-> Skipping x8 Validation (val_x8.log already exists)")

    # --- Step 5: Extract Stats ---
    vac_stats = extract_statistics(out_dir / "refine_vacuum.log", val_vac_log)
    x8_stats = extract_statistics(out_dir / "refine_x8.log", val_x8_log)

    return vac_stats, x8_stats

def main():
    """Concise CLI wrapper."""
    parser = argparse.ArgumentParser(description="Compare Phenix refinements between datasets.")
    parser.add_argument("pdb", help="Input PDB file")
    parser.add_argument("mtz_vacuum", help="Input Vacuum MTZ file")
    parser.add_argument("mtz_x8", help="Input x8 MTZ file")
    parser.add_argument("--flags", help="Optional R-free flags MTZ file", default=None)
    parser.add_argument("--outdir", help="Isolated directory for intermediate files", default=".")
    
    args = parser.parse_args()

    # Run the core function
    v_stats, x_stats = run_comparison(
        args.pdb, 
        args.mtz_vacuum, 
        args.mtz_x8, 
        flags_file=args.flags, 
        output_dir=args.outdir
    )
    
    # Output the table to the console
    if v_stats and x_stats:
        print_markdown_table(v_stats, x_stats)
    else:
        logger.error("Comparison failed. Table could not be generated.")

if __name__ == "__main__":
    main()