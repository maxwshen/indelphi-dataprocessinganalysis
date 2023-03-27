from __future__ import division
import _config, _lib, _data, _predict, _predict2
import sys, os, pickle
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
import matplotlib
matplotlib.use('Pdf')
from scipy.stats import pearsonr, entropy

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
# Run statistics
##
def calc_statistics(orig_df, exp, rate_model, bp_model, alldf_dict, rs):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  df = _lib.mh_del_subset(orig_df)
  df = _lib.indels_without_mismatches_subset(df)
  if sum(df['Count']) <= 1000:
    return
  
  ins_crit = (orig_df['Category'] == 'ins') & (orig_df['Length'] == 1)
  ins_df = orig_df[ins_crit]
  truncated_ins_d = defaultdict(list)
  for ins_base in list('ACGT'):
    crit = (ins_df['Inserted Bases'] == ins_base)
    tot_count = sum(ins_df[crit]['Count'])
    truncated_ins_d['Count'].append(tot_count)
    truncated_ins_d['Inserted Bases'].append(ins_base)
    truncated_ins_d['Category'].append('ins')
    truncated_ins_d['Length'].append(1)
  ins_df = pd.DataFrame(truncated_ins_d)
  df = df.append(ins_df, ignore_index = True)
  df['Frequency'] = _lib.normalize_frequency(df)

  _predict2.init_model()

  seq, cutsite = _lib.get_sequence_cutsite(orig_df)
  pred_df = _predict2.predict_mhdel(seq, cutsite)

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

  # Predict 1 bp genotype frequencies
  pred_1bpins_d = defaultdict(list)
  for ins_base in bp_model[fivebase]:
    freq = bp_model[fivebase][ins_base]
    freq *= rate_1bpins / (1 - rate_1bpins)

    pred_1bpins_d['Category'].append('ins')
    pred_1bpins_d['Length'].append(1)
    pred_1bpins_d['Inserted Bases'].append(ins_base)
    pred_1bpins_d['Predicted_Frequency'].append(freq)

  pred_1bpins_df = pd.DataFrame(pred_1bpins_d)
  pred_df = pred_df.append(pred_1bpins_df, ignore_index = True)
  pred_df['Predicted_Frequency'] /= sum(pred_df['Predicted_Frequency'])

  join_cols = ['Category', 'Genotype Position', 'Length', 'Inserted Bases']
  mdf = df.merge(pred_df, how = 'outer', on = join_cols)
  mdf['Frequency'].fillna(value = 0, inplace = True)
  mdf['Predicted_Frequency'].fillna(value = 0, inplace = True)
  obs = mdf['Frequency']
  pred = mdf['Predicted_Frequency']
  r = pearsonr(obs, pred)[0]
  alldf_dict['gt_r'].append(r)

  obs_entropy = entropy(obs) / np.log(len(obs))
  pred_entropy = entropy(pred) / np.log(len(pred))
  alldf_dict['obs entropy'].append(obs_entropy)
  alldf_dict['pred entropy'].append(pred_entropy)

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
    prefix = e_dir + 'gt_%s_%s' % (data_nm, rs)
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
