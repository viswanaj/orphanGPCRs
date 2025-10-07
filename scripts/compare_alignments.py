#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

from Bio import AlignIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator


def calculate_alignment_stats(alignment_fasta: str) -> Dict[str, float]:
    """Calculate basic statistics for an alignment."""
    alignment = AlignIO.read(alignment_fasta, "fasta")
    
    # Convert to MultipleSeqAlignment if needed
    if not isinstance(alignment, MultipleSeqAlignment):
        alignment = MultipleSeqAlignment(alignment)
    
    num_seqs = len(alignment)
    alignment_length = alignment.get_alignment_length()
    
    # Calculate identity percentage
    total_positions = num_seqs * alignment_length
    identical_positions = 0
    
    for i in range(alignment_length):
        column = alignment[:, i]
        # Count how many sequences have the same character as the first sequence
        first_char = column[0]
        if first_char != '-':  # Skip gaps
            identical_count = sum(1 for char in column if char == first_char and char != '-')
            identical_positions += identical_count
    
    identity_percentage = (identical_positions / total_positions) * 100 if total_positions > 0 else 0
    
    # Calculate gap percentage
    gap_count = sum(1 for record in alignment for char in record.seq if char == '-')
    gap_percentage = (gap_count / total_positions) * 100 if total_positions > 0 else 0
    
    return {
        'num_sequences': num_seqs,
        'alignment_length': alignment_length,
        'identity_percentage': identity_percentage,
        'gap_percentage': gap_percentage
    }


def compare_trees(newick_file1: str, newick_file2: str) -> Dict[str, float]:
    """Compare two phylogenetic trees using Robinson-Foulds distance."""
    try:
        tree1 = Phylo.read(newick_file1, "newick")
        tree2 = Phylo.read(newick_file2, "newick")
        
        # Calculate Robinson-Foulds distance
        rf_distance = tree1.compare(tree2)
        
        # Calculate normalized RF distance
        total_splits = len(tree1.get_nonterminals()) + len(tree2.get_nonterminals())
        normalized_rf = rf_distance / total_splits if total_splits > 0 else 0
        
        return {
            'robinson_foulds_distance': rf_distance,
            'normalized_rf_distance': normalized_rf
        }
    except Exception as e:
        print(f"Error comparing trees: {e}")
        return {'robinson_foulds_distance': float('inf'), 'normalized_rf_distance': float('inf')}


def run_alignment_comparison(input_fasta: str, base_output_dir: str) -> None:
    """Run all three alignment methods and compare results."""
    
    # Define output directories
    clustalo_dir = os.path.join(base_output_dir, "classA_alignment")
    mafft_dir = os.path.join(base_output_dir, "classA_alignment_mafft")
    muscle_dir = os.path.join(base_output_dir, "classA_alignment_muscle")
    
    print("=== Running Alignment Comparison ===")
    print(f"Input FASTA: {input_fasta}")
    print(f"Base output directory: {base_output_dir}")
    print()
    
    # Run Clustal Omega (EBI)
    print("1. Running Clustal Omega (EBI)...")
    try:
        subprocess.run([
            "python", "scripts/align_and_tree_ebi_clustalo.py",
            "--in_fasta", input_fasta,
            "--out_dir", clustalo_dir
        ], check=True)
        print("   ✓ Clustal Omega completed")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Clustal Omega failed: {e}")
        return
    
    # Run MAFFT
    print("2. Running MAFFT...")
    try:
        subprocess.run([
            "python", "scripts/align_and_tree_mafft.py",
            "--in_fasta", input_fasta,
            "--out_dir", mafft_dir
        ], check=True)
        print("   ✓ MAFFT completed")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ MAFFT failed: {e}")
        return
    
    # Run MUSCLE
    print("3. Running MUSCLE...")
    try:
        subprocess.run([
            "python", "scripts/align_and_tree_muscle.py",
            "--in_fasta", input_fasta,
            "--out_dir", muscle_dir
        ], check=True)
        print("   ✓ MUSCLE completed")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ MUSCLE failed: {e}")
        return
    
    print()
    print("=== Alignment Statistics ===")
    
    # Compare alignment statistics
    methods = [
        ("Clustal Omega", os.path.join(clustalo_dir, "classA_aligned.fasta")),
        ("MAFFT", os.path.join(mafft_dir, "classA_aligned.fasta")),
        ("MUSCLE", os.path.join(muscle_dir, "classA_aligned.fasta"))
    ]
    
    stats = {}
    for method_name, alignment_file in methods:
        if os.path.exists(alignment_file):
            stats[method_name] = calculate_alignment_stats(alignment_file)
            print(f"\n{method_name}:")
            print(f"  Sequences: {stats[method_name]['num_sequences']}")
            print(f"  Alignment length: {stats[method_name]['alignment_length']}")
            print(f"  Identity: {stats[method_name]['identity_percentage']:.2f}%")
            print(f"  Gaps: {stats[method_name]['gap_percentage']:.2f}%")
        else:
            print(f"\n{method_name}: Alignment file not found")
    
    print()
    print("=== Tree Comparison ===")
    
    # Compare trees
    tree_files = [
        ("Clustal Omega", os.path.join(clustalo_dir, "classA_tree.newick")),
        ("MAFFT", os.path.join(mafft_dir, "classA_tree.newick")),
        ("MUSCLE", os.path.join(muscle_dir, "classA_tree.newick"))
    ]
    
    for i, (method1, tree1) in enumerate(tree_files):
        for j, (method2, tree2) in enumerate(tree_files[i+1:], i+1):
            if os.path.exists(tree1) and os.path.exists(tree2):
                comparison = compare_trees(tree1, tree2)
                print(f"\n{method1} vs {method2}:")
                print(f"  Robinson-Foulds distance: {comparison['robinson_foulds_distance']}")
                print(f"  Normalized RF distance: {comparison['normalized_rf_distance']:.4f}")
            else:
                print(f"\n{method1} vs {method2}: Tree files not found")
    
    print()
    print("=== Summary ===")
    print("All three alignment methods have been run and compared.")
    print("Check the individual output directories for detailed results:")
    for method_name, _ in methods:
        print(f"  - {method_name}: {os.path.join(base_output_dir, method_name.lower().replace(' ', '_'))}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare Clustal Omega, MAFFT, and MUSCLE alignments.")
    ap.add_argument("--in_fasta", required=True, help="Input FASTA of Class A GPCRs")
    ap.add_argument("--out_dir", default=os.path.join(os.getcwd(), "data"), help="Base output directory")
    args = ap.parse_args()
    
    run_alignment_comparison(args.in_fasta, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
