# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, math, pickle, imp
sys.path.append('/cluster/mshen/')
import fnmatch
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
inp_dir = _config.OUT_PLACE + 'e_newgenotype/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)


##
# Load data
##
def load_lib_cas_data(nm, split):
  dtypes = {'Category': str, 'Count': float, 'Genotype Position': float, 'Indel with Mismatches': str, 'Ins Fivehomopolymer': str, 'Ins Template Length': float, 'Ins mh2': str, 'Ins p2': str, 'Inserted Bases': str, 'Length': float, 'Microhomology-Based': str, '_ExpDir': str, '_Experiment': str, '_Sequence Context': str, '_Cutsite': int}

  lib_exp = '051018_U2OS_+_LibA_preCas9'
  cas_exp = nm

  lib_data = pd.read_csv(inp_dir + '%s_genotypes_%s.csv' % (lib_exp, split), index_col = 0, dtype = dtypes)
  cas_data = pd.read_csv(inp_dir + '%s_genotypes_%s.csv' % (cas_exp, split), index_col = 0, dtype = dtypes)
  return lib_data, cas_data

##
# Individual adjustment
##
def build_new_cas(lib, cas):
  # Find shared indel events and place into a single DF.
  # Also use shared indel events to adjust counts in treatment
  lib_total = sum(lib['Count'])
  cas_total = sum(cas['Count'])
  if lib_total < 1000 or cas_total == 0:
    print lib_total
    return cas

  join_cols = list(cas.columns)
  join_cols.remove('_ExpDir')
  join_cols.remove('_Experiment')
  join_cols.remove('_Cutsite')
  join_cols.remove('_Sequence Context')
  join_cols.remove('Count')

  mdf = cas.merge(lib, how = 'outer', on = join_cols, suffixes = ('', '_control'))

  mdf['Count_control'].fillna(value = 0, inplace = True)
  mdf['Count'].fillna(value = 0, inplace = True)
  
  total_control = sum(mdf['Count_control'])
  total_treatment = sum(mdf['Count'])

  def control_adjust(row):
    ctrl_frac = row['Count_control'] / total_control
    treat_frac = row['Count'] / total_treatment
    if treat_frac == 0:
      return 0
    adjust_frac = (treat_frac - ctrl_frac) / treat_frac
    ans = max(row['Count'] * adjust_frac, 0)
    return ans

  new_counts = mdf.apply(control_adjust, axis = 1)
  mdf['Count'] = new_counts

  mdf = mdf[mdf['Count'] != 0]
  for col in mdf.columns:
    if '_control' in col:
      mdf = mdf.drop(labels = col, axis = 1)

  return mdf


##
# Main iterator
##
def control_adjustment(nm, split):
  print nm, split
  new_cas_data = pd.DataFrame()
  lib_data, cas_data = load_lib_cas_data(nm, split)
  for exp in set(cas_data['_Experiment']):
    cas = cas_data[cas_data['_Experiment'] == exp]
    lib = lib_data[lib_data['_Experiment'] == exp]
    new_cas = build_new_cas(lib, cas)
    new_cas_data = new_cas_data.append(new_cas, ignore_index = True)

  new_cas_data.to_csv(out_dir + '%s_genotypes_%s.csv' % (nm, split))

  return

##
# qsub
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print 'Generating qsub scripts...'
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  num_scripts = 0
  postcas9_nms = ['052218_U2OS_+_LibA_postCas9_rep1', '052218_U2OS_+_LibA_postCas9_rep2']

  for bc in postcas9_nms:
    for split in range(0, 2000, 100):
      command = 'python %s.py %s %s' % (NAME, bc, split)
      script_id = NAME.split('_')[0]

      # Write shell scripts
      sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, bc, split)
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


@util.time_dec
def main(nm = '', split = ''):
  print NAME  

  if nm == '' and split == '':
    gen_qsubs()
    return

  control_adjustment(nm, split)
  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(nm = sys.argv[1], split = sys.argv[2])
  else:
    main()
