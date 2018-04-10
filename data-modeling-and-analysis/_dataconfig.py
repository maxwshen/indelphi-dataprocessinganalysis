import pandas as pd
from collections import defaultdict
curr_dir = '/cluster/mshen/prj/mmej_figures/data/'

VO_T = pd.read_csv(curr_dir + 'overbeek_data.csv')

VO_DIR = '/cluster/mshen/prj/vanoverbeek/out/e10_control_adjustment/'

LIB_EXPS = defaultdict(dict)
with open(curr_dir + 'library_exps.txt') as f:
  for i, line in enumerate(f):
    w = line.strip().split(',')
    nm, fold, fold_nm = w[0], w[1], w[2]
    LIB_EXPS[nm]['fold'] = fold
    LIB_EXPS[nm]['exp_nm'] = fold_nm

VO_SEQS = dict()
with open(curr_dir + 'vo_seqs.fa') as f:
  for i, line in enumerate(f):
    if i % 2 == 0:
      header = line.strip()
      nm = header.replace('>', '')
    if i % 2 == 1:
      seq = line.strip()
      VO_SEQS[nm] = seq

NAMES_LIB1 = []
with open(curr_dir + 'names_lib1.txt') as f:
  for i, line in enumerate(f):
    NAMES_LIB1.append(line.strip())

TARGETS_LIB1 = []
with open(curr_dir + 'targets_lib1.txt') as f:
  for i, line in enumerate(f):
    TARGETS_LIB1.append(line.strip())

NAMES_DISLIB = []
with open(curr_dir + 'names_dislib.txt') as f:
  for i, line in enumerate(f):
    NAMES_DISLIB.append(line.strip())

TARGETS_DISLIB = []
with open(curr_dir + 'targets_dislib.txt') as f:
  for i, line in enumerate(f):
    TARGETS_DISLIB.append(line.strip())

DISLIB_WT = pd.read_csv(curr_dir + 'dislib_wt.csv', index_col = 0)

HIGHREP_DISLIB_EXPS = open(curr_dir + '_highrep_dislibexps.txt').readlines()
HIGHREP_DISLIB_EXPS = [int(s.strip()) for s in HIGHREP_DISLIB_EXPS]
HIGHREP_DISLIB_EXPS_NMS = [NAMES_DISLIB[idx] for idx in HIGHREP_DISLIB_EXPS]

HIGHREP_LIB1_EXPS = open(curr_dir + '_highrep_lib1exps.txt').readlines()
HIGHREP_LIB1_EXPS = [int(s.strip()) for s in HIGHREP_LIB1_EXPS]
HIGHREP_LIB1_EXPS_NMS = [NAMES_LIB1[idx] for idx in HIGHREP_LIB1_EXPS]
