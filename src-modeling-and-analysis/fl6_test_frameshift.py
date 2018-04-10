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
from scipy.stats import pearsonr, entropy
import e_ins_modeling

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
        ]

##
# Bae
##
def bae_scorer(s1, s2):
  assert len(s1) == len(s2)
  assert len(s1) > 0
  # Input: Two sequences, across a deletion
  MATCH = {'G': 2, 'C': 2, 'A': 1, 'T': 1}
  scores = []
  curr_score = 0
  len_match = 0
  for i in range(len(s1)):
    if s1[i] == s2[i]:
      curr_score += MATCH[s1[i]]
      len_match += 1
    else:
      if len_match > 1:
        scores.append(curr_score)
      curr_score = 0
      len_match = 0
  scores.append(curr_score)
  return scores

def bae_prediction(seq, cutsite):
  scores = []
  for del_len in range(1, 28+1):
    left = seq[cutsite - del_len : cutsite]
    right = seq[cutsite : cutsite + del_len]
    bs = bae_scorer(left, right)
    scores.append( math.exp(-del_len/20) * sum(bs) )
  pred = np.array(scores) / sum(scores)
  return pred

##
# Run statistics
##
def calc_statistics(orig_df, exp, rate_model, bp_model, alldf_dict, rs):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  df = _lib.mh_del_subset(orig_df)
  df = _lib.indels_without_mismatches_subset(df)
  if sum(df['Count']) <= 1000:
    return
  
  df = orig_df

  # Get observed frameshift rates
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



  # Predict
  _predict2.init_model()

  seq, cutsite = _lib.get_sequence_cutsite(orig_df)

  # Predict rate of 1 bp insertions
    # Featurize first
  del_score = _predict2.total_deletion_score(seq, cutsite)
  dlpred = _predict2.deletion_length_distribution(seq, cutsite)
  norm_entropy = entropy(dlpred) / np.log(len(dlpred))
  ohmapper = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
  fivebase = seq[cutsite - 1]
  onebp_features = ohmapper[fivebase] + [norm_entropy] + [del_score]
  onebp_features = np.array(onebp_features).reshape(1, -1)
  rate_1bpins = float(rate_model.predict(onebp_features))

  # Predict 1 bp frequency
  freq = rate_1bpins / (1 - rate_1bpins)
  pred = list(dlpred)
  pred.insert(0, freq)
  pred = np.array(pred) / sum(pred)

  pred_fs = {'+0': 0, '+1': 0, '+2': 0}
  pred_fs['+1'] += pred[0]
  for idx in range(1, len(pred)):
    del_len = idx
    fs = (-1 * del_len) % 3
    key = '+%s' % (int(fs))
    pred_fs[key] += pred[idx]

  # Bae predict
  bae_fs = {'+0': 0, '+1': 0, '+2': 0}
  bae_dlpred = bae_prediction(seq, cutsite)
  for idx in range(len(bae_dlpred)):
    del_len = idx + 1
    fs = (-1 * del_len) % 3
    key = '+%s' % (int(fs))
    bae_fs[key] += bae_dlpred[idx]

  for fs in ['+0', '+1', '+2']:
    alldf_dict['Frame'].append(fs)
    alldf_dict['Bae'].append(bae_fs[fs])
    alldf_dict['inDelphi'].append(pred_fs[fs])
    alldf_dict['Obs'].append(obs_fs[fs])

    alldf_dict['_Experiment'].append(exp)
    alldf_dict['rs'].append(rs)

  return alldf_dict




def prepare_statistics(data_nm):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  dataset = _data.load_dataset(data_nm, exp_subset = 'vo_spacers', exp_subset_col = 'Designed Name')
  if dataset is None:
    return

  e_dir = '/cluster/mshen/prj/mmej_figures/out/e_ins_modeling/'
  timer = util.Timer(total = 100)
  for rs in range(100):
    prefix = e_dir + 'len_%s_%s' % (data_nm, rs)
    test_exps = pickle.load(open(prefix + '_testexps.pkl'))
    rate_model = pickle.load(open(prefix + '_model.pkl'))
    bp_model = pickle.load(open(prefix + '_bp.pkl'))

    for exp in test_exps:
      df = dataset[exp]
      calc_statistics(df, exp, rate_model, bp_model, alldf_dict, rs)

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
