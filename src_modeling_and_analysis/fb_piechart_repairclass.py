from __future__ import division
import _config, _lib, _data
import sys, os, datetime, subprocess, math, pickle, imp
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
DEFAULT_INP_DIR = None
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
redo = False

##
# Going wide: experiments to analyze
##
exps = ['VO-spacers-HEK293-48h-controladj', 
        'VO-spacers-HCT116-48h-controladj',
        'VO-spacers-K562-48h-controladj',
        '0226-PRLmESC-Lib1-Cas9',
        '0226-PRLmESC-Dislib-Cas9',
        'Lib1-mES-controladj', 
        'DisLib-mES-controladj', 
        'DisLib-U2OS-controladj', 
        'cpf1-HEK-lenti-controladj',
        'cpf1-HEK-plasmid-controladj',
        'cpf1-HCT-plasmid-controladj',
        ]

##
# Run statistics
##
def calc_statistics(df, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  # Denominator is all dels
  counts = dict()
  cats = ['del', 'del_notatcut', 'ins', 'ins_notatcut', 'combination_indel', 'del_onecut']
  for cat in cats:
    counts[cat] = sum(df[df['Category'] == cat]['Count'])
  
  total = sum(counts.values())
  if total <= 1000:
    return

  for cat in counts:
    freq = counts[cat] / total
    alldf_dict[cat].append(freq)

  # subcategories
  criteria = (df['Category'] == 'del') & (df['Microhomology-Based'] == 'yes')
  alldf_dict['del_mh'].append(sum(df[criteria]['Count']) / total)

  criteria = (df['Category'] == 'del') & (df['Microhomology-Based'] != 'yes')
  alldf_dict['del_nomh'].append(sum(df[criteria]['Count']) / total)

  criteria = (df['Category'] == 'ins') & (df['Ins Template Length'] >= 4)
  alldf_dict['ins_templated'].append(sum(df[criteria]['Count']) / total)

  try:
    criteria = (df['Category'] == 'ins') & (df['Ins Fivehomopolymer'] == 'yes') & (df['Ins Template Length'] < 4)
    alldf_dict['ins_5hm'].append(sum(df[criteria]['Count']) / total)
  except:
    alldf_dict['ins_5hm'].append(0)

  try:
    criteria = (df['Category'] == 'ins') & (df['Ins Fivehomopolymer'] != 'yes') & (df['Ins Template Length'] < 4)
    alldf_dict['ins_other'].append(sum(df[criteria]['Count']) / total)
  except:
    alldf_dict['ins_other'].append(0)

  try:
    criteria = (df['Category'] == 'ins') & (df['Length'] == 1)
    alldf_dict['ins_1bp'].append(sum(df[criteria]['Count']) / total)
  except:
    alldf_dict['ins_1bp'].append(0)

  try:
    criteria = (df['Category'] == 'ins') & (df['Length'] != 1)
    alldf_dict['ins_not1bp'].append(sum(df[criteria]['Count']) / total)
  except:
    alldf_dict['ins_not1bp'].append(0)

  criteria = (df['Category'].isin(['del', 'del_onecut', 'del_notatcut']))
  alldf_dict['all_del'].append(sum(df[criteria]['Count']) / total)

  criteria = (df['Category'].isin(['ins', 'ins_notatcut']))
  alldf_dict['all_ins'].append(sum(df[criteria]['Count']) / total)

  alldf_dict['_Experiment'].append(exp)

  return alldf_dict

def prepare_statistics(data_nm):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  dataset = _data.load_dataset(data_nm)
  if dataset is None:
    return

  timer = util.Timer(total = len(dataset))
  # for exp in dataset.keys()[:100]:
  for exp in dataset.keys():
    df = dataset[exp]
    calc_statistics(df, exp, alldf_dict)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  alldf = pd.DataFrame(alldf_dict)
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
  for exp in exps:
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
