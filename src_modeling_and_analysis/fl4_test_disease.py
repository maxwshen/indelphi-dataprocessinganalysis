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

  try:
    df = _lib.mh_del_subset(orig_df)
  except:
    return
  df = _lib.indels_without_mismatches_subset(df)
  if sum(df['Count']) <= 1000:
    return
  df['Frequency'] = _lib.normalize_frequency(df)

  _predict2.init_model()

  seq, cutsite = _lib.get_sequence_cutsite(df)
  pred_df = _predict2.predict_mhdel(seq, cutsite)

  join_cols = ['Category', 'Genotype Position', 'Length']
  mdf = df.merge(pred_df, how = 'outer', on = join_cols)
  mdf['Frequency'].fillna(value = 0, inplace = True)
  mdf['Predicted_Frequency'].fillna(value = 0, inplace = True)

  # All crispr subset for finding total wt repair frequency
  csdf = _lib.crispr_subset(orig_df)
  csdf['Frequency'] = _lib.normalize_frequency(csdf)

  # Wildtype repair frequency
  dlwt = _config.d.DISLIB_WT
  row = dlwt[dlwt['name'] == exp].iloc[0]
  alldf_dict['_wt_repairable'].append(row['wt_repairable'])
  alldf_dict['_fs_repairable'].append(row['fs_repairable'])
  alldf_dict['_needed_fs'].append(row['fs'])
  for col in dlwt.columns:
    alldf_dict[col].append(row[col])

  if row['wt_repairable'] != 'yes':
    alldf_dict['_wt_obs_in_mhdels'].append(np.nan)
    alldf_dict['_wt_obs_in_allediting'].append(np.nan)
    alldf_dict['_wt_pred_in_mhdels'].append(np.nan)
    alldf_dict['_dl'].append(np.nan)
  else:
    dls = [int(s) for s in row['dls'].split(';')]
    gts = [int(s) for s in row['gts'].split(';')]

    obs_freq, pred_freq = 0, 0

    obs_freq_all = 0
    for dl, gt in zip(dls, gts):
      crit = (mdf['Length'] == dl) & (mdf['Genotype Position'] == gt)
      obs_freq += sum(mdf[crit]['Frequency'])
      pred_freq += sum(mdf[crit]['Predicted_Frequency'])

      crit = (csdf['Category'] == 'del') & (csdf['Length'] == dl) & (csdf['Genotype Position'] == gt) & (csdf['Indel with Mismatches'] != 'yes')
      obs_freq_all += sum(csdf[crit]['Frequency'])

    alldf_dict['_wt_obs_in_mhdels'].append(obs_freq)
    alldf_dict['_wt_obs_in_allediting'].append(obs_freq_all)
    alldf_dict['_wt_pred_in_mhdels'].append(pred_freq)
    alldf_dict['_dl'].append(set(dls).pop())

  # # Frameshift repair frequency
  # #  Currently does not incorporate 1bp insertions
  
  if row['fs_repairable'] != 'yes':
    alldf_dict['_fs_obs'].append(np.nan)
    alldf_dict['_fs_pred'].append(np.nan)
  else:
    # frameshifts recorded in +insertion orientation

    # Observed frameshifts in all crispr indels
    df = orig_df
    obs_fs = {'+0': 0, '+1': 0, '+2': 0}
    all_ins_lens = set(df[df['Category'].isin(['ins', 'ins_notatcut'])]['Length'])
    for ins_len in all_ins_lens:
      crit = (df['Category'].isin(['ins', 'ins_notatcut'])) & (df['Length'] == ins_len)
      fs = ins_len % 3
      count = sum(df[crit]['Count'])
      key = '+%s' % (int(fs))
      obs_fs[key] += count

    all_del_lens = set(df[df['Category'].isin(['del', 'del_notatcut'])]['Length'])
    for del_len in all_del_lens:
      crit = (df['Category'].isin(['del', 'del_notatcut'])) & (df['Length'] == del_len)
      fs = (-1*del_len) % 3
      count = sum(df[crit]['Count'])
      key = '+%s' % (int(fs))
      obs_fs[key] += count

    tot = sum(obs_fs.values())
    for key in obs_fs:
      obs_fs[key] /= tot

    # Predict frameshifts
    pred_fs = {'+0': 0, '+1': 0, '+2': 0}
    dlpred = _predict2.deletion_length_distribution(seq, cutsite)
    for idx, pred in enumerate(dlpred):
      del_len = idx + 1
      fs = (-1*del_len) % 3
      key = '+%s' % (int(fs))
      pred_fs[key] += pred

    # Get needed frameshift
    needed_fs = row['fs']
    # Map from -deletion (as recorded in data file) 
    # to +insertion orientation which is used here
    fs_orientation_mapper = {0: 0, 1: 2, 2: 1}
    needed_fs = fs_orientation_mapper[needed_fs]
    needed_fs_key = '+%s' % (needed_fs)

    obs_freq = obs_fs[needed_fs_key]
    pred_freq = pred_fs[needed_fs_key]
    alldf_dict['_fs_obs'].append(obs_freq)
    alldf_dict['_fs_pred'].append(pred_freq)

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
  alldf.sort_index(axis=1, inplace=True)
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
