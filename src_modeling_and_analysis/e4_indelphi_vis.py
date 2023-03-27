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
# Run statistics
##
def calc_statistics(orig_df, exp, rate_model, bp_model, alldf_dict, rs, data_nm):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  df = _lib.mh_del_subset(orig_df)
  df = _lib.indels_without_mismatches_subset(df)
  if sum(df['Count']) <= 1000:
    return
  
  obs_d = defaultdict(list)

  df = orig_df
  # Grab observed deletions, MH and MH-less
  for del_len in range(1, 59+1):
    crit = (df['Category'] == 'del') & (df['Indel with Mismatches'] != 'yes') & (df['Length'] == del_len)
    s = df[crit]

    mh_s = s[s['Microhomology-Based'] == 'yes']
    for idx, row in mh_s.iterrows():
      obs_d['Count'].append(row['Count'])
      obs_d['Genotype Position'].append(row['Genotype Position'])
      obs_d['Length'].append(row['Length'])
      obs_d['Category'].append('del')

    mhless_s = s[s['Microhomology-Based'] != 'yes']
    obs_d['Length'].append(del_len)
    obs_d['Count'].append(sum(mhless_s['Count']))
    obs_d['Genotype Position'].append('e')
    obs_d['Category'].append('del')

  obs_df = pd.DataFrame(obs_d) 

  # Grab observed 1 bp insertions
  ins_crit = (orig_df['Category'] == 'ins') & (orig_df['Length'] == 1) & (orig_df['Indel with Mismatches'] != 'yes')
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
  obs_df = obs_df.append(ins_df, ignore_index = True)

  obs_df['Frequency'] = _lib.normalize_frequency(obs_df)

  crispr_subset = _lib.crispr_subset(orig_df)
  frac_explained = sum(obs_df['Count']) / sum(crispr_subset['Count'])
  # print frac_explained
  # Save this for aggregate plotting
  alldf_dict['Fraction Explained'].append(frac_explained)

  # Predict MH dels and MH-less dels
  _predict2.init_model()
  seq, cutsite = _lib.get_sequence_cutsite(orig_df)
  pred_df = _predict2.predict_indels(seq, cutsite, 
                                     rate_model, bp_model)


  mdf = obs_df.merge(pred_df, how = 'outer', on = ['Category', 'Genotype Position', 'Inserted Bases', 'Length'])
  mdf['Frequency'].fillna(value = 0, inplace = True)
  mdf['Predicted_Frequency'].fillna(value = 0, inplace = True)
  r = pearsonr(mdf['Frequency'], mdf['Predicted_Frequency'])[0]

  # Merge observed and predicted, double check correlation
  # Store in a way that I can plot it.

  data_nm_out_dir = out_dir + data_nm + '/'
  util.ensure_dir_exists(data_nm_out_dir)
  exp_out_dir = data_nm_out_dir + exp + '/'
  util.ensure_dir_exists(exp_out_dir)
  out_fn = exp_out_dir + '%.3f.csv' % (r)
  mdf.to_csv(out_fn)

  # Store in alldf_dict
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
  # for rs in range(1):
    prefix = e_dir + 'len_%s_%s' % (data_nm, rs)
    test_exps = pickle.load(open(prefix + '_testexps.pkl'))
    rate_model = pickle.load(open(prefix + '_model.pkl'))
    bp_model = pickle.load(open(prefix + '_bp.pkl'))

    for exp in test_exps:
      df = dataset[exp]
      calc_statistics(df, exp, rate_model, bp_model, alldf_dict, rs, data_nm)

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
