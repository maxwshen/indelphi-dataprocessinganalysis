from __future__ import division
import _config, _lib, _data, _predict2
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

# Default params
DEFAULT_INP_DIR = None
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
redo = False

##
# Going wide: experiments to analyze
##
exps = ['Lib1-mES-controladj', 
        '0226-PRLmESC-Lib1-Cas9',
        'DisLib-mES-controladj', 
        'DisLib-U2OS-controladj', 
        'DisLib-HEK293T', 
        'DisLib-U2OS-HEK-Mixture', 
        '0226-PRLmESC-Dislib-Cas9',
        'VO-spacers-HEK293-48h-controladj', 
        'VO-spacers-HCT116-48h-controladj', 
        'VO-spacers-K562-48h-controladj',
        '0105-mESC-Lib1-Cas9-Tol2-BioRep2-r1-controladj',
        '0105-mESC-Lib1-Cas9-Tol2-BioRep3-r1-controladj',
        '1207-mESC-Dislib-Cas9-Tol2-Biorep1-r1-controladj'
        '1207-mESC-Dislib-Cas9-Tol2-Biorep1-r2-controladj'
        ]

##
# Run statistics
##
def calc_statistics(df, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  # Denominator is ins
  if sum(_lib.crispr_subset(df)['Count']) <= 1000:
    return

  editing_rate = sum(_lib.crispr_subset(df)['Count']) / sum(_lib.notnoise_subset(df)['Count'])
  alldf_dict['Editing Rate'].append(editing_rate)

  ins_criteria = (df['Category'] == 'ins') & (df['Length'] == 1) & (df['Indel with Mismatches'] != 'yes')
  ins_count = sum(df[ins_criteria]['Count'])

  del_criteria = (df['Category'] == 'del') & (df['Indel with Mismatches'] != 'yes')
  del_count = sum(df[del_criteria]['Count'])
  if del_count == 0:
    return
  alldf_dict['Ins1bp/Del Ratio'].append(ins_count / (del_count + ins_count))

  mhdel_crit = (df['Category'] == 'del') & (df['Indel with Mismatches'] != 'yes') & (df['Microhomology-Based'] == 'yes')
  mhdel_count = sum(df[mhdel_crit]['Count'])
  try:
    alldf_dict['Ins1bp/MHDel Ratio'].append(ins_count / (mhdel_count + ins_count))
  except ZeroDivisionError:
    alldf_dict['Ins1bp/MHDel Ratio'].append(0)

  ins_ratio = ins_count / sum(_lib.crispr_subset(df)['Count'])
  alldf_dict['Ins1bp Ratio'].append(ins_ratio)

  seq, cutsite = _lib.get_sequence_cutsite(df)
  fivebase = seq[cutsite - 1]
  alldf_dict['Fivebase'].append(fivebase)

  _predict2.init_model()
  del_score = _predict2.total_deletion_score(seq, cutsite)
  alldf_dict['Del Score'].append(del_score)

  dlpred = _predict2.deletion_length_distribution(seq, cutsite)
  from scipy.stats import entropy
  norm_entropy = entropy(dlpred) / np.log(len(dlpred))
  alldf_dict['Entropy'].append(norm_entropy)

  local_seq = seq[cutsite - 4 : cutsite + 4]
  gc = (local_seq.count('C') + local_seq.count('G')) / len(local_seq)
  alldf_dict['GC'].append(gc)

  if fivebase == 'A':
    fivebase_oh = np.array([1, 0, 0, 0])
  if fivebase == 'C':
    fivebase_oh = np.array([0, 1, 0, 0])
  if fivebase == 'G':
    fivebase_oh = np.array([0, 0, 1, 0])
  if fivebase == 'T':
    fivebase_oh = np.array([0, 0, 0, 1])
  alldf_dict['Fivebase_OH'].append(fivebase_oh)

  threebase = seq[cutsite]
  alldf_dict['Threebase'].append(threebase)
  if threebase == 'A':
    threebase_oh = np.array([1, 0, 0, 0])
  if threebase == 'C':
    threebase_oh = np.array([0, 1, 0, 0])
  if threebase == 'G':
    threebase_oh = np.array([0, 0, 1, 0])
  if threebase == 'T':
    threebase_oh = np.array([0, 0, 0, 1])
  alldf_dict['Threebase_OH'].append(threebase_oh)

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
