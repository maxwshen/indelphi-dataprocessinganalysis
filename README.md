# crispr-indelphi-dataprocessinganalysis

Data Processing Pipeline

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

### FAQ

##### Are the processed data available?

The fully processed data is available at https://doi.org/10.6084/m9.figshare.6838016, https://doi.org/10.6084/m9.figshare.6837959, https://doi.org/10.6084/m9.figshare.6837956, https://doi.org/10.6084/m9.figshare.6837953, and https://doi.org/10.6084/m9.figshare.6837947.

These data are derived from the raw reads using the data processing code in this github. Briefly, this involves demultiplexing the reads, performing sequence alignment, shifting sequence alignment into a standardized format and data quality filtering. The output is a table of all observed genotypes in a standardized format, which should be the appropriate starting point for many users interested in our data. Unless your desired analysis unambiguously requires starting from the raw data, I would recommend starting from the processed data we have provided.

##### What is the structure of the raw reads?

The shorter read 1 contains the gRNA and the longer read 2 contains the designed 55-bp target site. In the case of a single sequencing read, this corresponds to the designed 55-bp target site. Read 1 containing the gRNA is not used in data processing. The 55-bp target sites designed for each library is available in SupplementaryData.xlsx. 

In read 2, the 55-bp target site is placed after a constant sequence such as "TCCGTGCTGTAACGAAAGGATGGGTGCGACGCGTCAT". Immediately following the 55-bp target site is the read 2 sequencing primer.