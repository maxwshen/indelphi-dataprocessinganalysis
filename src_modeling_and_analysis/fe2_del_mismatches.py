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
def get_mismatches(polish_dir, data, srr_id = None):
  local_data = defaultdict(lambda: 0)

  for fn in os.listdir(polish_dir):
    if fn[:3] == 'del' and 'not' not in fn:
      dl = int(fn.replace('del', '').replace('.txt', ''))
      if dl > 27:
        continue
      del_fn = polish_dir + '/' + fn

      with open(del_fn) as f:
        for i, line in enumerate(f):
          if i % 4 == 0:
            if srr_id is None:
              count = int(line.replace('>', '').split('_')[0])
            else:
              w = line.replace('>', '').split('_')
              count, read_start = int(w[0]), int(w[1])
          if i % 4 == 1:
            read = line.strip()
          if i % 4 == 2:
            genome = line.strip()

            read_start_idx = read.index(read.replace('-', '')[:1])
            del_start_idx = read_start_idx + read[read_start_idx:].index('-')
            rr = read[del_start_idx:]
            del_len = rr.index(rr.replace('-', '')[:1])
            del_end_idx = del_start_idx + del_len


            left_rd10 = read[del_start_idx - 10 : del_start_idx]
            left_ge10 = genome[del_start_idx - 10 : del_start_idx]

            right_rd10 = read[del_end_idx : del_end_idx + 10]
            right_ge10 = genome[del_end_idx : del_end_idx + 10]

            # if not full length, skip.
            if len(left_rd10 + left_ge10 + right_rd10 + right_ge10) != 40:
              continue

            local_data['total'] += count

            # Count left
            for idx in range(len(left_rd10)):
              relative_pos = idx - 10
              rd_base = left_rd10[idx]
              ge_base = left_ge10[idx]
              if rd_base != '-' and ge_base != '-' and rd_base != ge_base:
                # if count > 100:
                  # print 'pt2'
                  # import code; code.interact(local=dict(globals(), **locals()))
                local_data[relative_pos] += count
              else:
                local_data[relative_pos] += 0

            # Count right
            for idx in range(len(right_rd10)):
              relative_pos = idx
              rd_base = right_rd10[idx]
              ge_base = right_ge10[idx]
              if rd_base != '-' and ge_base != '-' and rd_base != ge_base:
                local_data[relative_pos] += count
              else:
                local_data[relative_pos] += 0

  data['total'].append(local_data['total'])
  for pos in range(-10, 10+1):
    data[pos].append(local_data[pos])
    # if local_data[pos] / local_data['total'] >= 0.20:
      # import code; code.interact(local=dict(globals(), **locals()))

  return


##
# Build lib1 data
##
def build_library_data(out_dir, exp):
  print exp
  if exp in _config.d.LIB_EXPS:
    exp_dir = _config.d.LIB_EXPS[exp]['fold'] + 'c6_polish/' + _config.d.LIB_EXPS[exp]['exp_nm'] + '/'

  data = defaultdict(list)

  subexps = os.listdir(exp_dir)
  timer = util.Timer(total = len(subexps))
  for subexp in subexps:
    get_mismatches(exp_dir + subexp + '/', data)
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
  inp_dir = '/cluster/mshen/prj/vanoverbeek/out/b_polish/'

  if wildtype:
    castype = 'WT'
  else:
    castype = '48h'

  srrids = get_srr_ids(exp.replace('VO_', ''), castype)
  data = defaultdict(list)

  # Build data
  timer = util.Timer(total = len(srrids))
  for srr_id in srrids:
    get_mismatches(inp_dir + srr_id + '/', data, srr_id = srr_id)
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

  exp_pairs = [('VO_K562', 'VO_K562_WT'), 
              ('VO_HCT116', 'VO_HCT116_WT'),
              ('VO_HEK293', 'VO_HEK293_WT'),
              ('1027-HEK293T-Lib1-Cas9-Tol2-Biorep1-r1', '1027-HEK293T-Lib1-noCas9-Biorep1'),
              ('1027-HCT116-Lib1-Cas9-Tol2-Biorep1-r1', '1027-HCT116-Lib1-noCas9-Biorep1'),
              ('1207-mESC-Dislib-Cas9-Tol2-Biorep1-r1', '1207-mESC-Dislib-noCas9-Biorep1'),
              ('1207-mESC-Dislib-Cas9-Tol2-Biorep1-r2', '1207-mESC-Dislib-noCas9-Biorep1'),
              ('1215-U2OS-Dislib-Cas9-Tol2-Biorep1-r1', '1207-U2OS-Dislib-noCas9-Biorep1'),
              ('1027-mESC-Lib1-Cas9-Tol2-Biorep1-r1', '0226-mESC-Lib1-noCas9'),
              ('0105-mESC-Lib1-Cas9-Tol2-BioRep2-r1', '0226-mESC-Lib1-noCas9'),
              ('0105-mESC-Lib1-Cas9-Tol2-BioRep3-r1', '0226-mESC-Lib1-noCas9')]


  with PdfPages(out_dir + 'allplots.pdf', 'w') as pdf:
    for cas_exp, wt_exp in exp_pairs:
      cas_data = all_data[cas_exp]
      wt_data = all_data[wt_exp]

      cas_df = pd.DataFrame(cas_data)
      cas_df = cas_df[cas_df['total'] >= 1000]
      cas_df = cas_df.divide(cas_df['total'], axis = 'rows')
      cas_df = cas_df.drop('total', axis = 1)
      cas_melt = pd.melt(cas_df, value_vars = list(cas_df.columns), var_name = 'Relative Position', value_name = 'Frequency')
      cas_melt['Category'] = 'Treatment'

      wt_df = pd.DataFrame(wt_data)
      wt_df = wt_df[wt_df['total'] >= 1000]
      wt_df = wt_df.divide(wt_df['total'], axis = 'rows')
      wt_df = wt_df.drop('total', axis = 1)
      wt_melt = pd.melt(wt_df, value_vars = list(wt_df.columns), var_name = 'Relative Position', value_name = 'Frequency')
      wt_melt['Category'] = 'Control'

      if len(cas_df) <= 10:
        continue

      joined = cas_melt.append([wt_melt], ignore_index = True)

      sns.pointplot(x = 'Relative Position', y = 'Frequency', hue = 'Category', data = joined, dodge = True)
      plt.tight_layout()
      plt.title('%s (N=%s) vs. %s (N=%s)' % (cas_exp, len(cas_df[0]), wt_exp, len(wt_df[0])))
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
