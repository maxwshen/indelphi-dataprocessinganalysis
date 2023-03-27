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
def get_mh_sections(d):
  if d['Microhomology-Based'].dtype == float or d['Microhomology-Based'].dtype == bool:
    mhyes_section = d[d['Microhomology-Based'] == True]
    mhno_section = d[d['Microhomology-Based'] == False]
  else:
    mhyes_section = d[(d['Microhomology-Based'] == 'True') | (d['Microhomology-Based'] == '1.0') | (d['Microhomology-Based'] == True)]
    mhno_section = d[(d['Microhomology-Based'] == 'False') | (d['Microhomology-Based'] == '0.0') | (d['Microhomology-Based'] == False)]
  return mhyes_section, mhno_section

def get_indels_no_mismatches(d):
  ddtype = d['Indel with Mismatches'].dtype 
  if ddtype == float or ddtype == bool:
    return d[d['Indel with Mismatches'] == False]
  else:
    return d[(d['Indel with Mismatches'] == '0.0') | (d['Indel with Mismatches'] == 'False') | (d['Indel with Mismatches'] == False)]
  return

def get_count(d, count_col, category = ''):
  # treat nan = 0
  return np.nansum(d[d['Category'] == category][count_col])

def individual_piechart(d, data):
  count_col = 'Adjusted Count'
  if 'Adjusted Count' not in d.columns:
    count_col = 'Count'

  counts = dict()
  counts['del'] = get_count(d, count_col, category = 'del')
  counts['del_notatcut'] = get_count(d, count_col, category = 'del_notatcut')

  counts['ins'] = get_count(d, count_col, category = 'ins')
  counts['ins_notatcut'] = get_count(d, count_col, category = 'ins_notatcut')

  counts['combination_indel'] = get_count(d, count_col, category = 'combination_indel')

  mhyes_section, mhno_section = get_mh_sections(d[d['Category'] == 'del'])
  counts['del_mh_yes'] = np.nansum(mhyes_section[count_col])
  counts['del_mh_no'] = np.nansum(mhno_section[count_col])

  try:
    counts['ins_fivehomopolymer'] = np.nansum(d[(d['Category'] == 'ins') & (d['Ins Fivehomopolymer'] == 'yes')][count_col])
  except:
    counts['ins_fivehomopolymer'] = 0
  try:
    counts['ins_templated'] = np.nansum(d[(d['Category'] == 'ins') & (d['Ins Template Length'] >= 6)][count_col])
  except:
    counts['ins_templated'] = 0

  total = counts['del'] + counts['del_notatcut'] + counts['ins'] + counts['ins_notatcut'] + counts['combination_indel']
  if total < 500:
    return

  for category in counts:
    frac = counts[category] / total
    data[category].append(frac)  
  return


##
# Build lib1 data
##
def build_library_data(out_dir, exp):
  print exp
  # if exp in ['2k-mES-Cas9-Tol2']:
    # inp_dir = '/cluster/mshen/prj/mmej_manda2/out/2017-10-27/e_newgenotype/'
  if exp in ['Lib1-mES', 'Lib1-HCT116', 'Lib1-HEK293T', 'DisLib-U2OS', 'DisLib-mES', 'DisLib-HEK293T', 'DisLib-U2OS-HEK-Mixture', 'PRL-Lib1-mES', 'PRL-DisLib-mES']:
    inp_dir = '/cluster/mshen/prj/mmej_figures/out/b_individualize/'
    exp_dir = inp_dir + exp + '/'

  data = defaultdict(list)

  timer = util.Timer(total = len(os.listdir(exp_dir)))
  for fn in os.listdir(exp_dir):
    if not fnmatch.fnmatch(fn, '*csv'):
      continue

    csv_fn = exp_dir + fn
    d = pd.read_csv(csv_fn)
    individual_piechart(d, data)
    timer.update()

  # Pickle, convert defaultdict to regular dict
  picklable_data = dict()
  for key in data:
    picklable_data[key] = data[key]

  with open(out_dir + '%s.pkl' % (exp), 'w') as f:
    pickle.dump(picklable_data, f)
  return data

## 
# Build VO Data
##
def get_srr_ids(celltype):
  srrids = []
  for idx, row in T.iterrows():
    nm = row['Library_Name']
    if '48hr' in nm and celltype in nm:
      srrids.append(row['Run'])
  print 'Found %s srr ids for %s' % (len(srrids), celltype)
  return srrids

def build_vo_data(out_dir, exp):
  print exp
  inp_dir = '/cluster/mshen/prj/vanoverbeek/out/e10_control_adjustment/'

  srrids = get_srr_ids(exp.replace('VO_', ''))
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
  for exp in ['VO_K562', 'VO_HCT116', 'VO_HEK293']:
    if qsub_nm != '' and exp != qsub_nm:
      continue
    print exp
    data_fn = out_dir + '%s.pkl' % (exp)
    if not os.path.isfile(data_fn):
      data = build_vo_data(out_dir, exp)
    else:
      print 'Loading previously processed pickled data..'
      with open(data_fn) as f:
        data = pickle.load(f)
    all_data[exp] = data
    all_exps.append(exp)

  # Load our library data
  for exp in ['Lib1-mES', 'Lib1-HCT116', 'Lib1-HEK293T', 'DisLib-U2OS', 'DisLib-mES', 'DisLib-HEK293T', 'PRL-Lib1-mES', 'PRL-DisLib-mES']:
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

  for exp in all_exps:
    data = all_data[exp]
    print exp
    for category in data:
      print '\t%s: %s' % (category, np.mean(data[category]))

  import code; code.interact(local=dict(globals(), **locals()))

  with PdfPages(out_dir + 'allplots.pdf', 'w') as pdf:
    for exp in all_exps:
      data = all_data[exp]

      dfd = defaultdict(list)
      for gt_start in data:
        for gt_end in data[gt_start]:
          val = np.mean(data[gt_start][gt_end])
          dfd['Start'].append(gt_start)
          dfd['End'].append(gt_end)
          dfd['Value'].append(val)

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
  for exp in ['VO_K562', 'VO_HCT116', 'VO_HEK293', 'Lib1-mES', 'Lib1-HCT116', 'Lib1-HEK293T', 'DisLib-U2OS', 'DisLib-mES', 'DisLib-HEK293T', 'DisLib-U2OS-HEK-Mixture', 'PRL-Lib1-mES', 'PRL-DisLib-mES']:
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
