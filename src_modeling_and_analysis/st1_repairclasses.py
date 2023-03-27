from __future__ import division
import _config
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

T = _config.d.VO_T

##
# Process data in a single experiment
##
def get_count(d, category = ''):
  return sum(d[d['Category'] == category]['Count'])

def individual_piechart(d, data):
  total = sum(d['Count'])

  if total < 500:
    return

  ## Noise
  counts = dict()
  counts['homopolymer'] = get_count(d, category = 'homopolymer')
  counts['hasN'] = get_count(d, category = 'hasN')
  counts['pcr'] = get_count(d, category = 'pcr_recombination')
  counts['poormatches'] = get_count(d, category = 'poormatches')

  counts['del'] = get_count(d, category = 'del')
  counts['del_notatcut'] = get_count(d, category = 'del_notatcut')
  counts['del_notcrispr'] = get_count(d, category = 'del_notcrispr')

  counts['ins'] = get_count(d, category = 'ins')
  counts['ins_notatcut'] = get_count(d, category = 'ins_notatcut')
  counts['ins_notcrispr'] = get_count(d, category = 'ins_notcrispr')

  counts['combination_indel'] = get_count(d, category = 'combination_indel')
  counts['forgiven_indel'] = get_count(d, category = 'forgiven_indel')
  counts['forgiven_combination_indel'] = get_count(d, category = 'forgiven_combination_indel')
  counts['combination_indel_notcrispr'] = get_count(d, category = 'combination_indel_notcrispr')

  counts['other'] = get_count(d, category = 'other')
  counts['wildtype'] = get_count(d, category = 'wildtype')

  for category in counts:
    frac = counts[category] / total
    data[category].append(frac)  
  return


##
# Build lib1 data
##
def build_library_data(out_dir, exp):
  print exp
  if exp in _config.d.LIB_EXPS:
    exp_dir = _config.d.LIB_EXPS[exp]['fold'] + 'e_newgenotype/'

  data = defaultdict(list)

  csv_fn = exp_dir + _config.d.LIB_EXPS[exp]['exp_nm'] + '.csv'
  df = pd.read_csv(csv_fn)
  subexps = set(df['Experiment'])
  timer = util.Timer(total = len(subexps))

  for subexp in subexps:
    d = df[df['Experiment'] == subexp]
    individual_piechart(d, data)
    timer.update()

  # Pickle, convert defaultdict to regular dict
  picklable_data = dict()
  for key in data:
    if key not in picklable_data:
      picklable_data[key] = data[key]

  with open(out_dir + '%s.pkl' % (exp), 'w') as f:
    pickle.dump(picklable_data, f)
  return data

## 
# Build VO Data
##
def get_srr_ids(celltype, cas_type):
  srrids = []
  for idx, row in T.iterrows():
    nm = row['Library_Name']
    if cas_type in nm and celltype in nm:
      srrids.append(row['Run'])
  print 'Found %s srr ids for %s' % (len(srrids), celltype)
  return srrids

def build_vo_data(out_dir, exp, wildtype = False):
  print exp
  inp_dir = '/cluster/mshen/prj/vanoverbeek/out/c_genotype/'

  if wildtype:
    castype = 'WT'
  else:
    castype = '48h'

  srrids = get_srr_ids(exp.replace('VO_', ''), castype)
  data = defaultdict(list)

  # Build data
  timer = util.Timer(total = len(srrids))
  for srr_id in srrids:
    csv_fn = inp_dir + '%s.csv' % (srr_id)
    if os.path.isfile(csv_fn):
      d = pd.read_csv(csv_fn)
      if len(d) > 0:
        individual_piechart(d, data)
    timer.update()

  # Pickle, convert defaultdict to regular dict
  picklable_data = dict()
  for key in data:
    if key not in picklable_data:
      picklable_data[key] = data[key]

  with open(out_dir + '%s.pkl' % (exp), 'w') as f:
    pickle.dump(picklable_data, f)

  return data

##
# Load data
##
def prepare_data(inp_dir, out_dir, qsub_nm = ''):
  all_data = dict()
  all_exps = []

  # Load VO data
  for exp in ['VO_K562', 'VO_HCT116', 'VO_HEK293', 'VO_K562_WT', 'VO_HCT116_WT', 'VO_HEK293_WT']:
    if qsub_nm != '' and exp != qsub_nm:
      continue
    print exp
    data_fn = out_dir + '%s.pkl' % (exp)
    if not os.path.isfile(data_fn):
      data = build_vo_data(out_dir, exp, wildtype = bool('_WT' in exp))
    else:
      print 'Loading previously processed pickled data..'
      with open(data_fn) as f:
        data = pickle.load(f)
    all_data[exp] = data
    all_exps.append(exp)

  # Load our library data
  for exp in _config.d.LIB_EXPS.keys():
    if qsub_nm != '' and exp != qsub_nm:
      continue
    print exp
    data_fn = out_dir + '%s.pkl' % (exp)
    if not os.path.isfile(data_fn):
      data = build_library_data(out_dir, exp)
    else:
      print 'Loading previously processed pickled data..'
      with open(data_fn) as f:
        data = pickle.load(f)
    all_data['%s' % (exp)] = data
    all_exps.append('%s' % (exp))

  return all_data, all_exps

##
# Main plotter
##
def plotter(inp_dir, out_dir):
  # Frequency of deletions by length and MH basis.
  all_data, all_exps = prepare_data(inp_dir, out_dir)

  medians_df = defaultdict(list)
  for exp in all_exps:
    data = all_data[exp]
    medians_df['Experiment Set'].append(exp)
    medians_df['N'].append(len(data['hasN']))
    for category in data.keys():
      medians_df[category].append(np.median(data[category]))

  df = pd.DataFrame(medians_df)
  df.to_csv(out_dir + 'medians.csv')


  with PdfPages(out_dir + 'allplots.pdf', 'w') as pdf:
    for exp in all_exps:
      data = all_data[exp]

      df = pd.DataFrame(data)
      df = pd.melt(df, value_vars = list(df.columns), var_name = 'Category', value_name = 'Frequency')

      import code; code.interact(local=dict(globals(), **locals()))

      df = pd.DataFrame(dfd)
      df = df.pivot('Start', 'End', 'Value')

      sns.heatmap(df, robust = True)
      plt.title('%s, N = %s' % (exp, len(data[-1][0])))
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
  for exp in ['VO_K562', 'VO_HCT116', 'VO_HEK293', 'VO_K562_WT', 'VO_HCT116_WT', 'VO_HEK293_WT'] + _config.d.LIB_EXPS.keys():
    command = 'python %s.py %s' % (NAME, exp)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, exp)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -m e -wd %s %s' % (_config.SRC_DIR, sh_fn))

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print 'Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir)
  return


##
# Main
##
@util.time_dec
def main(inp_dir, out_dir, data_nm = ''):
  print NAME  
  util.ensure_dir_exists(out_dir)

  if data_nm == '':
    gen_qsubs()
    return

  if data_nm != 'plot':
    prepare_data(inp_dir, out_dir, qsub_nm = data_nm)
    return

  if data_nm == 'plot':
    plotter(inp_dir, out_dir)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/', data_nm = sys.argv[1])
  else:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')
