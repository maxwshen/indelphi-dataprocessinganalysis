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
exp_pairs = [('VO-spacers-HEK293-48h-controladj', 
                  'VO-spacers-K562-48h-controladj'),
             ('VO-spacers-HEK293-48h-controladj', 
                  'VO-spacers-HCT116-48h-controladj'),
             ('VO-spacers-HCT116-48h-controladj', 
                  'VO-spacers-K562-48h-controladj'),
             ('Lib1-mES-controladj',
                  'VO-spacers-HEK293-48h-controladj'),
             ('Lib1-mES-controladj',
                  'VO-spacers-HCT116-48h-controladj'),
             ('Lib1-mES-controladj',
                  'VO-spacers-K562-48h-controladj'),
             ('DisLib-mES-controladj',
                  'VO-spacers-HEK293-48h-controladj'),
             ('DisLib-mES-controladj',
                  'VO-spacers-HCT116-48h-controladj'),
             ('DisLib-mES-controladj',
                  'VO-spacers-K562-48h-controladj'),
             ('Lib1-HEK293T-controladj',
                  'VO-spacers-HEK293-48h-controladj'),
             ('Lib1-HEK293T-controladj',
                  'VO-spacers-HCT116-48h-controladj'),
             ('Lib1-HEK293T-controladj',
                  'VO-spacers-K562-48h-controladj'),
             ('Lib1-HEK293T-controladj-v2',
                  'VO-spacers-HEK293-48h-controladj'),
             ('Lib1-HEK293T-controladj-v2',
                  'VO-spacers-HCT116-48h-controladj'),
             ('Lib1-HEK293T-controladj-v2',
                  'VO-spacers-K562-48h-controladj'),
             ('Lib1-HCT116-controladj',
                  'VO-spacers-HEK293-48h-controladj'),
             ('Lib1-HCT116-controladj',
                  'VO-spacers-HCT116-48h-controladj'),
             ('Lib1-HCT116-controladj',
                  'VO-spacers-K562-48h-controladj'),
             ('DisLib-HEK293T',
                  'VO-spacers-HEK293-48h-controladj'),
             ('DisLib-HEK293T',
                  'VO-spacers-HCT116-48h-controladj'),
             ('DisLib-HEK293T',
                  'VO-spacers-K562-48h-controladj'),
             ('DisLib-HEK293T-v2',
                  'VO-spacers-HEK293-48h-controladj'),
             ('DisLib-HEK293T-v2',
                  'VO-spacers-HCT116-48h-controladj'),
             ('DisLib-HEK293T-v2',
                  'VO-spacers-K562-48h-controladj'),
             ('DisLib-U2OS-controladj',
                  'VO-spacers-HEK293-48h-controladj'),
             ('DisLib-U2OS-controladj',
                  'VO-spacers-HCT116-48h-controladj'),
             ('DisLib-U2OS-controladj',
                  'VO-spacers-K562-48h-controladj'),
             ('1027-HEK293T-Lib1-Cas9-Tol2-Biorep1-r1-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1207-HEK293T-Lib1-Cas9-Tol2-Biorep1-r2-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1207-HEK293T-Lib1-Cas9-Tol2-Biorep1-r3-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HEK2Lib-new-2000x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HEK2Lib-new-3000-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HEK2Lib-new-3000to1500x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HEK2Lib-new-3000to2000x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HEK2Lib-old-1500x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HEK2Lib-old-2000x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HEK2Lib-old-2500x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HEK293T-Dislib-Cas9-Tol2-Biorep1-r1', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('0105-HEK-Dislib-Cas9-Tol2-Biorep1-r2', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('0105-HEK-Dislib-Cas9-Tol2-Biorep1-r3', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('0105-U2OS-Dislib-Cas9-Tol2-Biorep1-r2-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-U2OS-Dislib-Cas9-Tol2-Biorep1-r1-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1027-HCT116-Lib1-Cas9-Tol2-Biorep1-r1-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HCT2kLib-new-1000x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HCT2kLib-new-1500x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HCT2k-old-1000x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
              ('1215-HCT2k-old-2000x-controladj', 
                  'VO-spacers-HEK293-48h-controladj'),
             ]

##
# Run statistics
##
def calc_statistics(df1, df2, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions
  df1 = _lib.crispr_subset(df1)
  df2 = _lib.crispr_subset(df2)

  if sum(df1['Count']) < 1000 or sum(df2['Count']) < 1000:
    return

  def get_r_from_subsets(d1, d2):
    if sum(d1['Count']) < 100 or sum(d2['Count']) < 100:
      return np.nan
    d1['Frequency'] = _lib.normalize_frequency(d1)
    d2['Frequency'] = _lib.normalize_frequency(d2)
    mdf = _lib.merge_crispr_events(d1, d2, '_1', '_2')
    return pearsonr(mdf['Frequency_1'], mdf['Frequency_2'])[0]

  # everything
  alldf_dict['all'].append(get_r_from_subsets(df1, df2)) 

  # All CRISPR dels
  d1 = _lib.del_subset(df1)
  d2 = _lib.del_subset(df2)
  alldf_dict['All del'].append(get_r_from_subsets(d1, d2))

  # Del at cut
  d1 = df1[df1['Category'] == 'del']
  d2 = df2[df2['Category'] == 'del']
  alldf_dict['del'].append(get_r_from_subsets(d1, d2))

  # MH dels
  d1 = df1[(df1['Category'] == 'del') & (df1['Microhomology-Based'] == 'yes')]
  d2 = df2[(df2['Category'] == 'del') & (df2['Microhomology-Based'] == 'yes')]
  alldf_dict['mh_del'].append(get_r_from_subsets(d1, d2))

  # MHless dels
  d1 = df1[(df1['Category'] == 'del') & (df1['Microhomology-Based'] == 'no')]
  d2 = df2[(df2['Category'] == 'del') & (df2['Microhomology-Based'] == 'no')]
  alldf_dict['nomh_del'].append(get_r_from_subsets(d1, d2))

  # Del not at cut
  d1 = df1[df1['Category'] == 'del_notatcut']
  d2 = df2[df2['Category'] == 'del_notatcut']
  alldf_dict['del_notatcut'].append(get_r_from_subsets(d1, d2))

  # All CRISPR ins
  d1 = _lib.ins_subset(df1)
  d2 = _lib.ins_subset(df2)
  alldf_dict['All ins'].append(get_r_from_subsets(d1, d2))

  # All ins at cutsite
  d1 = df1[df1['Category'] == 'ins']
  d2 = df2[df2['Category'] == 'ins']
  alldf_dict['ins'].append(get_r_from_subsets(d1, d2))

  # 1bp ins
  d1 = df1[(df1['Category'] == 'ins') & (df1['Length'] == 1)]
  d2 = df2[(df2['Category'] == 'ins') & (df2['Length'] == 1)]
  alldf_dict['ins_1bp'].append(get_r_from_subsets(d1, d2))

  # 2bp+ ins
  d1 = df1[(df1['Category'] == 'ins') & (df1['Length'] > 1)]
  d2 = df2[(df2['Category'] == 'ins') & (df2['Length'] > 1)]
  alldf_dict['ins_2bpplus'].append(get_r_from_subsets(d1, d2))

  # Ins not at cut
  d1 = df1[df1['Category'] == 'ins_notatcut']
  d2 = df2[df2['Category'] == 'ins_notatcut']
  alldf_dict['ins_notatcut'].append(get_r_from_subsets(d1, d2))

  alldf_dict['_Experiment'].append(exp)
  return

def prepare_statistics(data_nm1, data_nm2):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # In this case: Distribution of frequencies of indels for each position in 20 bp window around cutsite. Can plot mean, median, etc, difference, etc.
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  # If Library, subset VO spacers
  dataset1 = _data.load_dataset(data_nm1, exp_subset = 'vo_spacers', exp_subset_col = 'Designed Name')
  dataset2 = _data.load_dataset(data_nm2, exp_subset = 'vo_spacers', exp_subset_col = 'Designed Name')
  if dataset1 is None or dataset2 is None:
    return

  # Find shared exps and iterate through them, passing both shared exps together to calc_statistics
  shared_exps = set(dataset1.keys()) & set(dataset2.keys())
  if len(shared_exps) == 0:
    print 'ERROR: No shared exps'
    import code; code.interact(local=dict(globals(), **locals()))

  timer = util.Timer(total = len(shared_exps))
  for exp in shared_exps:
    d1 = dataset1[exp]
    d2 = dataset2[exp]
    calc_statistics(d1, d2, exp, alldf_dict)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  alldf = pd.DataFrame(alldf_dict)
  return alldf


##
# Load statistics from csv, or calculate 
##
def load_statistics(data_nm1, data_nm2):
  print data_nm1, data_nm2
  stats_csv_fn = out_dir + '%s_%s.csv' % (data_nm1, data_nm2)
  if not os.path.isfile(stats_csv_fn) or redo:
    print 'Running statistics from scratch...'
    stats_csv = prepare_statistics(data_nm1, data_nm2)
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
      exp1, exp2 = exp_p
      stats = load_statistics(exp1, exp2)

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
    exp1, exp2 = exp_p
    command = 'python %s.py %s %s redo' % (NAME, exp1, exp2)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, exp1, exp2)
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
def main(data_nm1 = '', data_nm2 = '', redo_flag = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)

  if redo_flag == 'redo':
    global redo
    redo = True

  if data_nm1 == '' and data_nm2 == '':
    gen_qsubs()
    return

  if data_nm1 == 'plot':
    plot()

  else:
    load_statistics(data_nm1, data_nm2)

  return


if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(data_nm1 = sys.argv[1])
  elif len(sys.argv) == 3:
    main(data_nm1 = sys.argv[1], data_nm2 = sys.argv[2])
  elif len(sys.argv) == 4:
    main(data_nm1 = sys.argv[1], data_nm2 = sys.argv[2], redo_flag = sys.argv[3])
  else:
    main()
