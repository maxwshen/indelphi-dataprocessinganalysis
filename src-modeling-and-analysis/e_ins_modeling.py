from __future__ import division
import _config, _lib, _data, _predict, _predict2
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
import random
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
from mylib import compbio
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# Default params
DEFAULT_INP_DIR = '/cluster/mshen/prj/mmej_manda2/out/2017-10-27/mb_grab_exons/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'

##
# Functions
##
def convert_oh_string_to_nparray(input):
    input = input.replace('[', '').replace(']', '')
    nums = input.split(' ')
    return np.array([int(s) for s in nums])

def featurize(rate_stats, Y_nm):
    fivebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Fivebase_OH']])
    ent = np.array(rate_stats['Entropy']).reshape(len(rate_stats['Entropy']), 1)
    del_scores = np.array(rate_stats['Del Score']).reshape(len(rate_stats['Del Score']), 1)
    gc = np.array(rate_stats['GC']).reshape(len(rate_stats['GC']), 1)
    print ent.shape, fivebases.shape, del_scores.shape

    # Y = np.array(rate_stats['Ins1bp/Del Ratio'])
    Y = np.array(rate_stats[Y_nm])
    # Y = np.array(rate_stats['Ins1bp Ratio'])

    X = np.concatenate((fivebases, ent, del_scores), axis = 1)
    feature_names = ['A', 'C', 'G', 'T', 'Entropy', 'DelScore']
    print 'Num. samples: %s, num. features: %s' % X.shape

    return X, Y

def generate_models(X, Y, exps, bp_stats, rs, cv_nm):
  ans = train_test_split(X, Y, exps,
                         test_size = 0.20, 
                         random_state = rs)
  X_train, X_test, Y_train, Y_test, EXPS_train, EXPS_test = ans
  with open(out_dir + '%s_testexps.pkl' % (cv_nm), 'w') as f:
    pickle.dump(EXPS_test, f)

  # Train rate model
  model = KNeighborsRegressor(weights = 'distance')
  model.fit(X_train, Y_train)
  with open(out_dir + '%s_model.pkl' % (cv_nm), 'w') as f:
    pickle.dump(model, f)

  # Obtain bp stats
  bp_model = dict()
  bp_crit = (bp_stats['_Experiment'].isin(EXPS_train))
  train_bp_stats = bp_stats[bp_crit]
  ins_bases = ['A frac', 'C frac', 'G frac', 'T frac']
  t_melt = pd.melt(train_bp_stats, 
                   id_vars = ['Base'], 
                   value_vars = ins_bases, 
                   var_name = 'Ins Base', 
                   value_name = 'Fraction')
  for base in list('ACGT'):
    bp_model[base] = dict()
    mean_vals = []
    for ins_base in ins_bases:
      crit = (t_melt['Base'] == base) & (t_melt['Ins Base'] == ins_base)
      mean_vals.append(float(np.mean(t_melt[crit])))
    for bp, freq in zip(list('ACGT'), mean_vals):
      bp_model[base][bp] = freq / sum(mean_vals)

  with open(out_dir + '%s_bp.pkl' % (cv_nm), 'w') as f:
    pickle.dump(bp_model, f)
  return

def setup_cross_validation():
  import fi2_ins_ratio
  import fk_1bpins

  exps = ['VO-spacers-HEK293-48h-controladj', 
              'VO-spacers-K562-48h-controladj',
              'VO-spacers-HCT116-48h-controladj']
  
  for exp in exps:
    rate_stats = fi2_ins_ratio.load_statistics(exp)
    rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
    bp_stats = fk_1bpins.load_statistics(exp)
    exps = rate_stats['_Experiment']

    # Task 1 : Deletion Length
    X, Y = featurize(rate_stats, 'Ins1bp/Del Ratio')
    for rs in range(100):
      cv_nm = 'len_%s_%s' % (exp, rs)
      generate_models(X, Y, exps, bp_stats, rs, cv_nm)

    # Task 2 : Deletion Genotypes
    X, Y = featurize(rate_stats, 'Ins1bp/MHDel Ratio')
    for rs in range(100):
      cv_nm = 'gt_%s_%s' % (exp, rs)
      generate_models(X, Y, exps, bp_stats, rs, cv_nm)

  return

##
# Main
##
@util.time_dec
def main(data_nm = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)


  setup_cross_validation()

  return


if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(data_nm = sys.argv[1])
  else:
    main()
