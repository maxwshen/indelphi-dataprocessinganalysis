# 
from __future__ import division
import _config, _lib, _data
import sys, os, fnmatch, datetime, subprocess, math, pickle, imp
sys.path.append('/cluster/mshen/')
import fnmatch
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
DEFAULT_INP_DIR = None
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'

##
# Sequence featurization
##
def get_gc_frac(seq):
  return (seq.count('C') + seq.count('G')) / len(seq)

def find_microhomologies(left, right):
  start_idx = max(len(right) - len(left), 0)
  mhs = []
  mh = [start_idx]
  for idx in range(min(len(right), len(left))):
    if left[idx] == right[start_idx + idx]:
      mh.append(start_idx + idx + 1)
    else:
      mhs.append(mh)
      mh = [start_idx + idx +1]
  mhs.append(mh)
  return mhs

##
# Main featurizer
##
def featurize(orig_df):
  seq, cutsite = _lib.get_sequence_cutsite(orig_df)
  mh_lens, gc_fracs, del_lens, freqs = [], [], [], []
  dl_freqs = []

  DELLEN_LIMIT = 60

  df = _lib.mh_del_subset(orig_df)
  df = _lib.indels_without_mismatches_subset(df)
  df = df[df['Length'] <= DELLEN_LIMIT]

  if sum(df['Count']) < 1000:
    return None

  criteria = (orig_df['Category'] == 'del') & (orig_df['Length'] <= 28)
  s = orig_df[criteria]
  s['Frequency'] = _lib.normalize_frequency(s)
  for del_len in range(1, 28+1):
    dl_freq = sum(s[s['Length'] == del_len]['Frequency'])
    dl_freqs.append(dl_freq)

  df['Frequency'] = _lib.normalize_frequency(df)  


  for del_len in range(1, DELLEN_LIMIT + 1):
    left = seq[cutsite - del_len : cutsite]
    right = seq[cutsite : cutsite + del_len]

    mhs = find_microhomologies(left, right)
    for mh in mhs:
      mh_len = len(mh) - 1
      if mh_len > 0:
        gtpos = max(mh)

        s = cutsite - del_len + gtpos - mh_len
        e = s + mh_len
        mh_seq = seq[s : e]
        gc_frac = get_gc_frac(mh_seq)

        criteria = (df['Length'] == del_len) & (df['Genotype Position'] == gtpos)
        freq = sum(df[criteria]['Frequency'])

        mh_lens.append(mh_len)
        gc_fracs.append(gc_frac)
        del_lens.append(del_len)
        freqs.append(freq)



  return mh_lens, gc_fracs, del_lens, freqs, dl_freqs

##
# main 
##
def prepare_library_dataset(dataset, featurized_data):
  print 'Assumes library data'

  good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs = featurized_data

  timer = util.Timer(total = len(dataset))
  for exp in dataset.keys():
    df = dataset[exp]
    ans = featurize(df)
    if ans is None:
      continue
    mh_len, gc_frac, del_len, freq, dl_freq = ans
    good_exps.append(exp)
    mh_lengths.append(mh_len)
    gc_fracs.append(gc_frac)
    del_lens.append(del_len)
    freqs.append(freq)
    dl_freqs.append(dl_freq)
    timer.update()

  print 'Found %s good exps' % (len(good_exps))
  return

def init_featurized_data():
  good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs = [], [], [], [], [], []
  all_data = [good_exps, mh_lengths, gc_fracs, del_lens, freqs, dl_freqs]
  return all_data

def pickle_featurized_data(featurized_data, nm):
  print 'Pickling..'
  with open(out_dir + '%s.pkl' % (nm), 'w') as f:
    pickle.dump(featurized_data, f)
  print 'Done'
  return

##
# Dataset
##
def prepare_dataset_try1():
  dataset_nm = 'dataset_try1'
  print 'Preparing %s' % (dataset_nm)

  featurized_data = init_featurized_data()

  # Components...
  dataset = _data.load_dataset('DisLib-mES-controladj', 
                               exp_subset = 'longdup_series',
                               exp_subset_col = 'Designed Name')
  prepare_library_dataset(dataset, featurized_data)

  dataset = _data.load_dataset('Lib1-mES-controladj')

  # Remove VO spacers from lib 1
  for vo_spacer_idx in range(1872, 1961+1):
    vo_spacer_exp = str(vo_spacer_idx)
    del dataset[vo_spacer_exp]
  print len(dataset)
  prepare_library_dataset(dataset, featurized_data)

  pickle_featurized_data(featurized_data, dataset_nm)
  return

##
# Dataset
##
def prepare_dataset_try2():
  dataset_nm = 'dataset_try2'
  print 'Preparing %s' % (dataset_nm)

  featurized_data = init_featurized_data()

  # Load dislib, longdups
  dataset = _data.load_dataset('DisLib-mES-controladj', 
                               exp_subset = 'longdup_series',
                               exp_subset_col = 'Designed Name')
  prepare_library_dataset(dataset, featurized_data)

  # Load dislib, clin data
  dataset = _data.load_dataset('DisLib-mES-controladj', 
                               exp_subset = 'clin',
                               exp_subset_col = 'Designed Name')

  # Remove data with iterated editing
  dlwt = _config.d.DISLIB_WT
  for idx, row in dlwt.iterrows():
    if row['wt_repairable'] == 'iterwt':
      del dataset[row['name']]
  print len(dataset)
  prepare_library_dataset(dataset, featurized_data)

  # Load Lib1 data
  dataset = _data.load_dataset('Lib1-mES-controladj')

  # Remove VO spacers from lib 1
  for vo_spacer_idx in range(1872, 1961+1):
    vo_spacer_exp = str(vo_spacer_idx)
    del dataset[vo_spacer_exp]
  print len(dataset)
  prepare_library_dataset(dataset, featurized_data)

  pickle_featurized_data(featurized_data, dataset_nm)
  return


##
# Dataset
##
def prepare_dataset_try3():
  dataset_nm = 'dataset_try3'
  print 'Preparing %s' % (dataset_nm)

  featurized_data = init_featurized_data()

  # Components...
  dataset = _data.load_dataset('DisLib-mES-controladj', 
                               exp_subset = 'longdup_series',
                               exp_subset_col = 'Designed Name')
  for exp in dataset.keys():
    if exp not in _config.d.HIGHREP_DISLIB_EXPS_NMS:
      del dataset[exp]
  prepare_library_dataset(dataset, featurized_data)

  dataset = _data.load_dataset('Lib1-mES-controladj')

  # Remove VO spacers from lib 1
  for vo_spacer_idx in range(1872, 1961+1):
    vo_spacer_exp = str(vo_spacer_idx)
    del dataset[vo_spacer_exp]

  # Remove low rep spacers from lib1
  for exp in dataset.keys():
    if int(exp) not in _config.d.HIGHREP_LIB1_EXPS:
      del dataset[exp]

  print len(dataset)
  prepare_library_dataset(dataset, featurized_data)

  pickle_featurized_data(featurized_data, dataset_nm)
  return


##
# Dataset
##
def prepare_dataset_try4():
  dataset_nm = 'dataset_try4'
  print 'Preparing %s' % (dataset_nm)

  featurized_data = init_featurized_data()

  # Load dislib, longdups
  dataset = _data.load_dataset('DisLib-mES-controladj', 
                               exp_subset = 'longdup_series',
                               exp_subset_col = 'Designed Name')
  for exp in dataset.keys():
    if exp not in _config.d.HIGHREP_DISLIB_EXPS_NMS:
      del dataset[exp]
  prepare_library_dataset(dataset, featurized_data)

  # Load dislib, clin data
  dataset = _data.load_dataset('DisLib-mES-controladj', 
                               exp_subset = 'clin',
                               exp_subset_col = 'Designed Name')

  # Remove data with iterated editing
  dlwt = _config.d.DISLIB_WT
  for idx, row in dlwt.iterrows():
    if row['wt_repairable'] == 'iterwt':
      del dataset[row['name']]
  for exp in dataset.keys():
    if exp not in _config.d.HIGHREP_DISLIB_EXPS_NMS:
      del dataset[exp]
  print len(dataset)
  prepare_library_dataset(dataset, featurized_data)

  # Load Lib1 data
  dataset = _data.load_dataset('Lib1-mES-controladj')

  # Remove VO spacers from lib 1
  for vo_spacer_idx in range(1872, 1961+1):
    vo_spacer_exp = str(vo_spacer_idx)
    del dataset[vo_spacer_exp]
  # Remove low rep spacers from lib1
  for exp in dataset.keys():
    if int(exp) not in _config.d.HIGHREP_LIB1_EXPS:
      del dataset[exp]

  print len(dataset)
  prepare_library_dataset(dataset, featurized_data)

  pickle_featurized_data(featurized_data, dataset_nm)
  return


##
# Main
##
@util.time_dec
def main(data_nm = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)

  # prepare_dataset_try1()
  # prepare_dataset_try2()
  prepare_dataset_try3()
  # prepare_dataset_try4()

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(data_nm = sys.argv[1])
  else:
    main()
