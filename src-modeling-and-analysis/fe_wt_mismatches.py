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
def set_master_expected_cutsite(srr_id):
  srr_row = T[T['Run'] == srr_id]
  if len(srr_row) == 0:
    return False, None
  cloc = str(srr_row['chromosome_loc']).split()[1]
  genome_build = str(cloc.split('_')[0])
  cloc = cloc.split('_')[1]
  chrm = str(cloc.split(':')[0])
  start = int(cloc.split(':')[1].split('-')[0])
  end = int(cloc.split(':')[1].split('-')[1])

  tool = '/cluster/mshen/tools/2bit/twoBitToFa'
  twobit_db = '/cluster/mshen/tools/2bit/%s.2bit' % (genome_build)
  twobit_start = start - 1
  command = '%s -seq=%s -start=%s -end=%s %s temp_%s.fa; cat temp_%s.fa' % (tool, chrm, twobit_start, end, twobit_db, srr_id, srr_id)
  query = subprocess.check_output(command, shell = True)
  genome = ''.join(query.split()[1:]).upper()
  if genome[:2] == 'CC' and genome[-2:] != 'GG':
    master_expected_cutsite = start + 7
  elif genome[:2] != 'CC' and genome[-2:] == 'GG':
    master_expected_cutsite = start + 23 - 6
  elif genome[:2] == 'CC' and genome[-2:] == 'GG':
    # If both CC and GG are available, default to GG. 
    # Three out of 96 spacers have both CC/GG, all three are GG.
    master_expected_cutsite = start + 23 - 6
  else:
    print 'ERROR: Expected gRNA lacks NGG on both strands'
    sys.exit(0)
  return master_expected_cutsite

def get_mismatches(wt_fn, data, srr_id = None):
  if srr_id is not None:
    master_cutsite = set_master_expected_cutsite(srr_id)
  else:
    master_cutsite = len('TCCGTGCTGTAACGAAAGGATGGGTGCGACGCGTCAT') + 27

  local_data = defaultdict(lambda: 0)

  with open(wt_fn) as f:
    for i, line in enumerate(f):
      if i % 4 == 0:
        if srr_id is None:
          count = int(line.replace('>', '').split('_')[0])
          cutsite = master_cutsite
        else:
          w = line.replace('>', '').split('_')
          count, read_start = int(w[0]), int(w[1])
          cutsite = master_cutsite - read_start
      if i % 4 == 1:
        read = line.strip()
      if i % 4 == 2:
        genome = line.strip()

        if srr_id is None:
          genome_start_idx = genome.index(genome.replace('-', '')[:1])
          cutsite += genome_start_idx

        rd20 = read[cutsite - 10 : cutsite + 10 + 1]
        ge20 = genome[cutsite - 10 : cutsite + 10 + 1]

        local_data['total'] += count
        for idx in range(len(rd20)):
          relative_pos = idx - 10
          if rd20[idx] != ge20[idx] and rd20[idx] in list('ACGT') and ge20[idx] in list('ACGT'):
            local_data[relative_pos] += count
          else:
            local_data[relative_pos] += 0

  data['total'].append(local_data['total'])
  for pos in range(-10, 10+1):
    data[pos].append(local_data[pos])

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
    wt_fn = exp_dir + subexp + '/wildtype.txt'
    if os.path.isfile(wt_fn):
      get_mismatches(wt_fn, data)

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
    wt_fn = inp_dir + '%s/wildtype.txt' % (srr_id)
    if os.path.isfile(wt_fn):
      get_mismatches(wt_fn, data, srr_id = srr_id)

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
      plt.title('%s vs. %s' % (cas_exp, wt_exp))
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
