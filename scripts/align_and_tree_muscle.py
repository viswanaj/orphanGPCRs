#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional

from Bio import AlignIO, Phylo
from Bio.Align.Applications import MuscleCommandline
from Bio.Phylo.Applications import PhymlCommandline


def run_muscle_alignment(input_fasta: str, output_fasta: str) -> None:
    """Run MUSCLE alignment on input FASTA file."""
    muscle_cline = MuscleCommandline(
        input=input_fasta,
        out=output_fasta,
        clw=True,  # Output in CLUSTALW format
        maxiters=100,  # Maximum number of iterations
        diags=True,  # Use diagonal optimization
        sv=True,  # Use distance measures
        maxhours=1.0  # Maximum time in hours
    )
    
    print("Running MUSCLE alignment...")
    stdout, stderr = muscle_cline()
    
    if stderr:
        print(f"MUSCLE stderr: {stderr}")
    
    # MUSCLE outputs in CLUSTALW format, convert to FASTA
    if os.path.exists(output_fasta):
        alignment = AlignIO.read(output_fasta, "clustal")
        AlignIO.write(alignment, output_fasta, "fasta")


def create_guide_tree(alignment_fasta: str, output_newick: str) -> None:
    """Create a guide tree from the alignment using UPGMA method."""
    # Read the alignment
    alignment = AlignIO.read(alignment_fasta, "fasta")
    
    # Create a distance matrix
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    from Bio.Align import MultipleSeqAlignment
    
    # Convert to MultipleSeqAlignment if needed
    if not isinstance(alignment, MultipleSeqAlignment):
        alignment = MultipleSeqAlignment(alignment)
    
    # Calculate distance matrix using BLOSUM62
    calculator = DistanceCalculator('blosum62')
    dm = calculator.get_distance(alignment)
    
    # Build tree using UPGMA
    constructor = DistanceTreeConstructor(calculator, 'upgma')
    tree = constructor.build_tree(alignment)
    
    # Write Newick format
    Phylo.write(tree, output_newick, "newick")


def create_phylogenetic_tree(alignment_fasta: str, output_phylotree: str) -> None:
    """Create a phylogenetic tree using PhyML (if available) or fallback to distance method."""
    try:
        # Try PhyML first
        phyml_cline = PhymlCommandline(
            input=alignment_fasta,
            datatype='aa',
            model='WAG',
            bootstrap=100,
            alpha='e',
            search='BEST'
        )
        
        print("Running PhyML phylogenetic analysis...")
        phyml_cline()
        
        # PhyML creates files with specific extensions
        base_name = Path(alignment_fasta).stem
        phyml_tree = f"{base_name}_phyml_tree.txt"
        
        if os.path.exists(phyml_tree):
            # Copy the PhyML tree to our output location
            with open(phyml_tree, 'r') as f:
                tree_content = f.read()
            with open(output_phylotree, 'w') as f:
                f.write(tree_content)
            
            # Clean up PhyML output files
            for ext in ['_phyml_tree.txt', '_phyml_stats.txt', '_phyml_tree.xml']:
                cleanup_file = f"{base_name}{ext}"
                if os.path.exists(cleanup_file):
                    os.remove(cleanup_file)
        else:
            # Fallback to distance method
            print("PhyML not available, using distance method...")
            create_guide_tree(alignment_fasta, output_phylotree)
            
    except Exception as e:
        print(f"PhyML failed: {e}")
        print("Falling back to distance method...")
        create_guide_tree(alignment_fasta, output_phylotree)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Align Class A GPCRs via MUSCLE and produce alignment and tree.")
    ap.add_argument("--in_fasta", required=True, help="Input FASTA of Class A GPCRs")
    ap.add_argument("--out_dir", default=os.path.join(os.getcwd(), "data", "classA_alignment_muscle"), help="Output directory")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_fasta = args.in_fasta
    output_fasta = out_dir / "classA_aligned.fasta"
    output_newick = out_dir / "classA_tree.newick"
    output_phylotree = out_dir / "classA_tree.phylotree"

    # Run MUSCLE alignment
    run_muscle_alignment(input_fasta, str(output_fasta))
    print("MUSCLE alignment completed.")

    # Create guide tree
    print("Creating guide tree...")
    create_guide_tree(str(output_fasta), str(output_newick))
    print("Guide tree created.")

    # Create phylogenetic tree
    print("Creating phylogenetic tree...")
    create_phylogenetic_tree(str(output_fasta), str(output_phylotree))
    print("Phylogenetic tree created.")

    print(f"Wrote alignment and trees to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
