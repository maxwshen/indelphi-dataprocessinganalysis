#!/usr/bin/env python
from __future__ import division
import _config
import sys, os, datetime, subprocess, math, pickle, imp
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
DEFAULT_INP_DIR = _config.OUT_PLACE + 'e_newgenotype/'
NAME = util.get_fn(__file__)

T = _config.d.TABLE

##
# Load data
##
def find_control(srr_id):
  nm = T[T['Run'] == srr_id]['Library_Name'].iloc[0]
  
  # Get features
  mtss = False
  if 'MTSS' in nm:
    mtss = True
  celltype = ''
  for ct in ['K562', 'HCT116', 'HEK293']:
    if ct in nm:
      celltype = ct
  hg = ''
  loc = ''
  for w in nm.split('_'):
    if w[:2] == 'hg':
      hg = w
    if w[:3] == 'chr':
      loc = w
  pkcs = False
  if 'pkcs_treated' in nm:
    pkcs = True

  # Search for control matching features
  control_srr_id = None
  for idx, row in T.iterrows():
    cand_nm = row['Library_Name']
    if 'WT' not in cand_nm and 'untreated' not in cand_nm:
      continue
    if hg in cand_nm and loc in cand_nm and celltype in cand_nm:
      if mtss and 'MTSS' not in cand_nm:
        continue
      if pkcs and 'pkcs_untreated' not in cand_nm:
        continue
      control_srr_id = row['Run']
      break

  if control_srr_id is None:
    return False

  return control_srr_id

def load_lib_cas_data(inp_dir, srr_id):
  lib_srr_id = find_control(srr_id)
  print 'Control srr_id: %s' % (lib_srr_id)
  if lib_srr_id is False:
    return False, False

  if not os.path.isfile(inp_dir + '%s.csv' % (srr_id)):
    print 'Bad fn %s' % (srr_id)
    return False, False

  dtypes = {'Category': str, 'Count': float, 'Genotype Position': float, 'Indel with Mismatches': str, 'Ins Fivehomopolymer': str, 'Ins Template Length': float, 'Ins mh2': str, 'Ins p2': str, 'Inserted Bases': str, 'Length': float, 'Microhomology-Based': str, '_ExpDir': str, '_Experiment': str, '_Sequence Context': str, '_Cutsite': int}


  lib_data = pd.read_csv(inp_dir + '%s.csv' % (lib_srr_id), index_col = 0, dtype = dtypes)
  cas_data = pd.read_csv(inp_dir + '%s.csv' % (srr_id), index_col = 0, dtype = dtypes)
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
def control_adjustment(inp_dir, out_dir, srr_id):
  print srr_id
  lib, cas = load_lib_cas_data(inp_dir, srr_id)

  if lib is False or cas is False:
    return

  new_cas = build_new_cas(lib, cas)

  new_cas.to_csv(out_dir + '%s.csv' % (srr_id))
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
  for idx in range(3696622, 3702820 + 1, 62):
    start = idx
    end = start + 61
    command = 'python %s.py none %s %s' % (NAME, start, end)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, start)
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


def is_control(srr_id):
  try:
    nm = T[T['Run'] == srr_id]['Library_Name'].iloc[0]
  except IndexError:
    return 'badnm'
  if 'WT' in nm or 'untreated' in nm:
    return True
  return False

##
# Main
##
@util.time_dec
def main(inp_dir, out_dir, srr_id = '', start = 'none', end = 'none'):
  print NAME  
  util.ensure_dir_exists(out_dir)

  # Function calls
  if srr_id == '' and start == 'none' and end == 'none':
    gen_qsubs()
    return

  if srr_id != '' and start == 'none' and end == 'none':
    if is_control(srr_id):
      print 'is control'
      return
    control_adjustment(inp_dir, out_dir, srr_id)
    return

  start, end = int(start), int(end)
  timer = util.Timer(total = end - start + 1)
  for idnum in range(start, end + 1):
    srr_id = 'SRR%s' % (idnum)
    ans = is_control(srr_id)
    if ans is False:
      control_adjustment(inp_dir, out_dir, srr_id)
    timer.update()
  
    
  return out_dir


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/', srr_id = sys.argv[1], start = sys.argv[2], end = sys.argv[3])
  else:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')
