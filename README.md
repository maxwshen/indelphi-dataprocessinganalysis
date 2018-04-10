# crispr-indelphi-dataprocessinganalysis

Processing Chain of Input/Output for Library data:

b_alignment:
- Input: Barcoded fasta files
- Output: Alignments in text file. Format: 4 lines per alignment: [>header, line1, line2, blank]
- Note: Parses read1 and performs LSH to identify the designed sequence context for alignment with read2.

c6_polish:
- Input: Above
- Output: Alignments in text files in a folder, with filenames corresponding to alignment category. Same format as above.
- Note: This script is different for VO and library data for several reasons (for example, free end gaps are disallowed in library data but allowed in VO data, requiring different processing approaches). For VO data, the input is text alignments converted from the SAM format.

e_newgenotype:
- Input: Above
- Output: CSV / pandas dataframe recording all alignment events and their counts.
- Note: Uniquely labels microhomology deletion genotypes using the largest delta-position that corresponds to the genotype in sequence alignment.

e10_control_adjustment:
- Input: A treatment CSV and control CSV for a single sequence context
- Output: Treatment CSV with control-adjusted counts
- Note: Subtracts the counts (adjusted for read-depth) of control events from treatment events to a floor of zero.

Data are then in an appropriate format for modeling and analysis.

Experiments with multiple replicates are combined using the \_data.py script in /src-modeling-analysis.
