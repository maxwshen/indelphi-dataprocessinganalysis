from __future__ import division
import _config, _lib, _data, _predict, _predict2
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
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
exps = ['DisLib-mES-controladj', 
        'DisLib-HEK293T', 
        'DisLib-U2OS-controladj', 
        '0226-PRLmESC-Dislib-Cas9',
        '1207-mESC-Dislib-Cas9-Tol2-Biorep1-r1-controladj',
        '1207-mESC-Dislib-Cas9-Tol2-Biorep1-r2-controladj',
        ]

##
# Run statistics
##
def calc_statistics(orig_df, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  df = _lib.crispr_subset(orig_df)
  if sum(df['Count']) <= 1000:
    return
  df['Frequency'] = _lib.normalize_frequency(df)

  # Wildtype repair frequency
  dlwt = _config.d.DISLIB_WT
  row = dlwt[dlwt['name'] == exp].iloc[0]
  if row['wt_repairable'] != 'yes':
    alldf_dict['wt_obs'].append(np.nan)
    alldf_dict['dl'].append(np.nan)
  else:
    dls = [int(s) for s in row['dls'].split(';')]
    gts = [int(s) for s in row['gts'].split(';')]

    obs_freq = 0
    for dl, gt in zip(dls, gts):
      crit = (df['Length'] == dl) & (df['Genotype Position'] == gt)
      obs_freq += sum(df[crit]['Frequency'])

    alldf_dict['wt_obs'].append(obs_freq)
    alldf_dict['dl'].append(set(dls).pop())

  alldf_dict['_Experiment'].append(exp)
  return alldf_dict

def prepare_statistics(data_nm):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  dataset = _data.load_dataset(data_nm, exp_subset = 'clin', exp_subset_col = 'Designed Name')
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
# nohups
##
def gen_nohups():
  # Generate qsub shell scripts and commands for easy parallelization
  print 'Generating nohup scripts...'
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  nh_commands = []

  num_scripts = 0
  for exp in exps:
    script_id = NAME.split('_')[0]
    command = 'nohup python -u %s.py %s redo > nh_%s_%s.out &' % (NAME, exp, script_id, exp)
    nh_commands.append(command)

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(nh_commands))

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
    gen_nohups()
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
