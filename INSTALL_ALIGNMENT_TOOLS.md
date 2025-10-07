# Installation Guide for Alignment Tools

This project now includes three alignment methods. The original Clustal Omega uses the EBI web service, but MAFFT and MUSCLE require local installation.

## Required Tools

### 1. MAFFT
MAFFT is a multiple sequence alignment program for amino acid or nucleotide sequences.

**Installation on macOS:**
```bash
# Using Homebrew
brew install mafft

# Or using conda
conda install -c bioconda mafft
```

**Installation on Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install mafft

# Or using conda
conda install -c bioconda mafft
```

**Installation on Windows:**
- Download from: https://mafft.cbrc.jp/alignment/software/
- Add to PATH or place executable in project directory

### 2. MUSCLE
MUSCLE is a multiple sequence alignment program for protein sequences.

**Installation on macOS:**
```bash
# Using Homebrew
brew install muscle

# Or using conda
conda install -c bioconda muscle
```

**Installation on Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install muscle

# Or using conda
conda install -c bioconda muscle
```

**Installation on Windows:**
- Download from: https://www.drive5.com/muscle/
- Add to PATH or place executable in project directory

### 3. PhyML (Optional)
PhyML is used for maximum likelihood phylogenetic tree construction.

**Installation on macOS:**
```bash
# Using Homebrew
brew install phyml

# Or using conda
conda install -c bioconda phyml
```

**Installation on Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install phyml

# Or using conda
conda install -c bioconda phyml
```

## Verification

After installation, verify the tools are available:

```bash
# Check MAFFT
mafft --version

# Check MUSCLE
muscle -version

# Check PhyML (optional)
phyml --version
```

## Usage

Once installed, you can run the alignment comparison:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run comparison (using Class A GPCRs as example)
python scripts/compare_alignments.py --in_fasta data/classes/gpcrs_human_reviewed_A.fasta

# Or run individual alignment methods:
python scripts/align_and_tree_mafft.py --in_fasta data/classes/gpcrs_human_reviewed_A.fasta
python scripts/align_and_tree_muscle.py --in_fasta data/classes/gpcrs_human_reviewed_A.fasta
```

### Dating a Tree (Strict Molecular Clock)

You can estimate node ages (in years) from a Newick tree assuming a strict molecular clock with either a known substitution rate or a calibration on total tree age:

```bash
# Option 1: Use a substitution rate (subs/site/year)
python scripts/date_tree_strict_clock.py --in_newick data/classA_alignment/classA_tree.newick --rate 1e-9 --out_csv data/classA_alignment/classA_tree_node_ages.csv

# Option 2: Calibrate so that the root age equals a given number of years
python scripts/date_tree_strict_clock.py --in_newick data/classA_alignment/classA_tree.newick --tree_age_years 5e8 --out_csv data/classA_alignment/classA_tree_node_ages.csv
```

Assumptions:
- Branch lengths are proportional to substitutions per site.
- Strict clock (constant rate across the tree).
- If using `--tree_age_years`, the root-to-tip path length defines the scaling.

## Notes

- The scripts will fall back to distance-based tree construction if PhyML is not available
- All three methods produce the same output format for easy comparison
- Output directories are separate to avoid conflicts between methods

### Plotting GPCR Age Histogram with Orphan Marks

You can plot a histogram of tip ages with orphan GPCRs highlighted as red rug marks. You must provide a rate or a root calibration.

```bash
# Using a substitution rate (subs/site/year), x-axis in million years
python scripts/plot_gpcr_age_histogram.py \
  --in_newick data/classA_alignment/classA_tree.newick \
  --rate 1e-9 \
  --units myr \
  --infer_orphans_by_regex \
  --out_png data/classA_alignment/gpcr_age_hist.png \
  --out_svg data/classA_alignment/gpcr_age_hist.svg

# Or with an explicit orphan list (one gene symbol per line)
python scripts/plot_gpcr_age_histogram.py \
  --in_newick data/classA_alignment/classA_tree.newick \
  --tree_age_years 5e8 \
  --orphans_list data/orphan_genes.txt \
  --units myr
```

By default, the script can infer orphan receptors as gene symbols matching `^GPR\d+$` if `--infer_orphans_by_regex` is provided. You can supply an explicit orphan list to override or augment this.
