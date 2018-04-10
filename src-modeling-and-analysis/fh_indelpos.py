from __future__ import division
import _config, _lib, _data
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
from mylib import compbio
import pandas as pd
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import pearsonr

# Default params
DEFAULT_INP_DIR = None
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
redo = False

##
# Going wide: experiments to analyze
##
exp_pairs = [('VO-spacers-HEK293-48h', 'VO-spacers-HEK293-WT'),
             ('VO-spacers-HCT116-48h', 'VO-spacers-HCT116-WT'),
             ('VO-spacers-K562-48h', 'VO-spacers-K562-WT'),
             ('Lib1-mES', '0226-mESC-Lib1-noCas9'),
             ('Lib1-HEK293T', '1027-HEK293T-Lib1-noCas9-Biorep1'),
             ('Lib1-HCT116', '1027-HCT116-Lib1-noCas9-Biorep1'),
             ('DisLib-mES', '1207-mESC-Dislib-noCas9-Biorep1'),
             ('DisLib-U2OS', '1207-U2OS-Dislib-noCas9-Biorep1')
             ]

##
# Run statistics
##
def calc_statistics(df, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  # Denominator is all non-noise categories
  df = _lib.notnoise_subset(df)
  df['Frequency'] = _lib.normalize_frequency(df)
  if sum(df['Frequency']) == 0:
    return

  # Consider only deletions, anywhere
  del_df = _lib.del_subset(df)
  # Get left side
  for del_pos in range(-10, -1 + 1):
    total_freq = sum(del_df[del_df['Genotype Position'] == del_pos]['Frequency'])
    alldf_dict[str(del_pos)].append(total_freq)

  # Get right side
  for del_pos in range(1, 10 + 1):
    criteria = (del_df['Genotype Position'] - del_df['Length'] == del_pos)
    total_freq = sum(del_df[criteria]['Frequency'])
    alldf_dict[str(del_pos)].append(total_freq)

  editing_rate = sum(_lib.crispr_subset(df)['Frequency']) / sum(df['Frequency'])
  alldf_dict['Editing Rate'].append(editing_rate)
  alldf_dict['_Experiment'].append(exp)

  # Test alternative hypothesis: is asymmetry actually meaningful? If -1 arises from 0gt and sequencing mismatch, and +1 arises from Ngt and sequencing mismatch, then we should see asymmetry in 0gt vs Ngt.
  def detect_0gt_microhomology(row):
    # Returns a frequency
    if row['Category'] != 'del':
      return 0
    cutsite = int(row['_Cutsite'])
    seq = row['_Sequence Context']
    gt_pos = int(row['Genotype Position'])
    del_len = int(row['Length'])

    left = seq[cutsite - del_len : cutsite]
    right = seq[cutsite : cutsite + del_len]
    if len(left) != len(right):
      return 0

    mhs = []
    mh = [0]
    for idx, (c1, c2) in enumerate(zip(left, right)):
      if c1 == c2:
        mh.append(idx+1)
      else:
        mhs.append(mh)
        mh = [idx+1]
    mhs.append(mh)

    for mh in mhs:
      if gt_pos in mh:
        if 0 in mh:
          return row['Frequency']
    return 0

  freq_0gt = sum(del_df.apply(detect_0gt_microhomology, axis = 1))
  alldf_dict['0gt Frequency'].append(freq_0gt)

  criteria = (del_df['Genotype Position'] - del_df['Length'] == 0)
  freq_Ngt = sum(del_df[criteria]['Frequency'])
  alldf_dict['Ngt Frequency'].append(freq_Ngt)
  return

def prepare_statistics(data_nm):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # In this case: Distribution of frequencies of indels for each position in 20 bp window around cutsite. Can plot mean, median, etc, difference, etc.
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  dataset = _data.load_dataset(data_nm)
  if dataset is None:
    return

  timer = util.Timer(total = len(dataset))
  for exp in dataset:
    df = dataset[exp]
    calc_statistics(df, exp, alldf_dict)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  alldf = pd.DataFrame(alldf_dict)
  col_order = ['_Experiment', 'Editing Rate', '0gt Frequency', 'Ngt Frequency', '-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
  if len(col_order) != len(alldf.columns):
    print 'ERROR: Will drop columns'
  alldf = alldf[col_order]
  return alldf


##
# Load statistics from csv, or calculate 
##
def load_statistics(data_nm):
  print data_nm
  stats_csv_fn = out_dir + '%s.csv' % (data_nm)
  if not os.path.isfile(stats_csv_fn) or redo:
    print 'Running statistics from scratch...'
    stats_csv = prepare_statistics(data_nm)
    stats_csv.to_csv(stats_csv_fn)
  else:
    print 'Getting statistics from file...'
    stats_csv = pd.read_csv(stats_csv_fn, index_col = 0)
  print 'Done'
  return stats_csv

##
# Plotters
##
def plot():
  # Frequency of deletions by length and MH basis.
  from scipy.stats import pearsonr
  
  with PdfPages(out_dir + '%s_plots.pdf' % (NAME), 'w') as pdf:
    for exp_p in exp_pairs:
      treatment_exp, control_exp = exp_p
      treatment_stats = load_statistics(treatment_exp)
      control_stats = load_statistics(control_exp)

      for del_pos in range(-10, 0) + range(1, 11):
        r, p_val = pearsonr(list(treatment_stats['Editing Rate']), list(treatment_stats[str(del_pos)]))
        print 'Correlation between editing rate and position %s: %s, p-value: %s' % (del_pos, r, p_val) 

      pos_cols = [str(s) for s in range(-10, 0)] + [str(s) for s in range(1, 11)]
      treatment_stats_melted = pd.melt(treatment_stats, value_vars = pos_cols, var_name = 'Position', value_name = 'Frequency')
      control_stats_melted = pd.melt(control_stats, value_vars = pos_cols, var_name = 'Position', value_name = 'Frequency')

      sns.pointplot(x = 'Position', y = 'Frequency', data = treatment_stats_melted, order = pos_cols, label = 'Treatment', color = 'r')
      sns.pointplot(x = 'Position', y = 'Frequency', data = control_stats_melted, order = pos_cols, label = 'Control', color = 'b')
      plt.title('%s vs. %s: Deletions by position' % (treatment_exp, control_exp))
      plt.tight_layout()
      pdf.savefig()
      plt.close()


  return


##
# qsubs
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print 'Generating qsub scripts...'
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  num_scripts = 0
  for exp_p in exp_pairs:
    for exp in exp_p:
      command = 'python %s.py %s redo' % (NAME, exp)
      script_id = NAME.split('_')[0]

      # Write shell scripts
      sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, exp)
      with open(sh_fn, 'w') as f:
        f.write('#!/bin/bash\n%s\n' % (command))
      num_scripts += 1

      # Write qsub commands
      qsub_commands.append('qsub -m e -V -wd %s %s' % (_config.SRC_DIR, sh_fn))

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print 'Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir)
  return


##
# Main
##
@util.time_dec
def main(data_nm = '', redo_flag = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)

  if redo_flag == 'redo':
    global redo
    redo = True

  if data_nm == '':
    gen_qsubs()
    return

  if data_nm == 'plot':
    plot()

  else:
    load_statistics(data_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(data_nm = sys.argv[1])
  elif len(sys.argv) == 3:
    main(data_nm = sys.argv[1], redo_flag = sys.argv[2])
  else:
    main()
