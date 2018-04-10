from __future__ import division
import _config
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
import numpy as np
from collections import defaultdict
from mylib import util
from mylib import compbio
import pandas as pd

# Expected data types for dataframes
csv_dtypes = {'Category': str, 'Count': float, 'Genotype Position': float, 'Indel with Mismatches': str, 'Ins Fivehomopolymer': str, 'Ins Template Length': float, 'Ins mh2': str, 'Ins p2': str, 'Inserted Bases': str, 'Length': float, 'Microhomology-Based': str, '_ExpDir': str, '_Experiment': str, '_Sequence Context': str, '_Cutsite': int}

memoized = dict()

D = pd.read_csv(_config.DATA_DIR + 'DataTable.csv', dtype = {'Name': str, 'Library': str, 'Memoization': str, 'Local_Name': str, 'Alignment_Directory': str, 'Data_Directory': str})

exp_subset_dict = {'vo_spacers': ['overbeek_spacer_1', 'overbeek_spacer_10', 'overbeek_spacer_11', 'overbeek_spacer_12', 'overbeek_spacer_13', 'overbeek_spacer_14', 'overbeek_spacer_15', 'overbeek_spacer_16', 'overbeek_spacer_17', 'overbeek_spacer_18', 'overbeek_spacer_19', 'overbeek_spacer_2', 'overbeek_spacer_20', 'overbeek_spacer_21', 'overbeek_spacer_22', 'overbeek_spacer_23', 'overbeek_spacer_24', 'overbeek_spacer_25', 'overbeek_spacer_26', 'overbeek_spacer_27', 'overbeek_spacer_28', 'overbeek_spacer_29', 'overbeek_spacer_3', 'overbeek_spacer_30', 'overbeek_spacer_31', 'overbeek_spacer_32', 'overbeek_spacer_33', 'overbeek_spacer_34', 'overbeek_spacer_35', 'overbeek_spacer_36', 'overbeek_spacer_37', 'overbeek_spacer_38', 'overbeek_spacer_39', 'overbeek_spacer_4', 'overbeek_spacer_40', 'overbeek_spacer_41', 'overbeek_spacer_42', 'overbeek_spacer_43', 'overbeek_spacer_44', 'overbeek_spacer_45', 'overbeek_spacer_46', 'overbeek_spacer_47', 'overbeek_spacer_48', 'overbeek_spacer_49', 'overbeek_spacer_5', 'overbeek_spacer_50', 'overbeek_spacer_51', 'overbeek_spacer_52', 'overbeek_spacer_53', 'overbeek_spacer_54', 'overbeek_spacer_55', 'overbeek_spacer_56', 'overbeek_spacer_57', 'overbeek_spacer_58', 'overbeek_spacer_59', 'overbeek_spacer_6', 'overbeek_spacer_60', 'overbeek_spacer_61', 'overbeek_spacer_62', 'overbeek_spacer_63', 'overbeek_spacer_64', 'overbeek_spacer_65', 'overbeek_spacer_66', 'overbeek_spacer_67', 'overbeek_spacer_68', 'overbeek_spacer_69', 'overbeek_spacer_7', 'overbeek_spacer_70', 'overbeek_spacer_71', 'overbeek_spacer_72', 'overbeek_spacer_73', 'overbeek_spacer_74', 'overbeek_spacer_75', 'overbeek_spacer_76', 'overbeek_spacer_77', 'overbeek_spacer_78', 'overbeek_spacer_79', 'overbeek_spacer_8', 'overbeek_spacer_80', 'overbeek_spacer_81', 'overbeek_spacer_82', 'overbeek_spacer_83', 'overbeek_spacer_84', 'overbeek_spacer_85', 'overbeek_spacer_86', 'overbeek_spacer_87', 'overbeek_spacer_88', 'overbeek_spacer_89', 'overbeek_spacer_9', 'overbeek_spacer_90', 'overbeek_spacer_91', 'overbeek_spacer_92', 'overbeek_spacer_93', 'overbeek_spacer_94', 'overbeek_spacer_95', 'overbeek_spacer_96'],
  'longdup_series': ['longdup_7_r1', 'longdup_8_r1', 'longdup_9_r1', 'longdup_10_r1', 'longdup_11_r1', 'longdup_12_r1', 'longdup_13_r1', 'longdup_14_r1', 'longdup_15_r1', 'longdup_16_r1', 'longdup_17_r1', 'longdup_18_r1', 'longdup_19_r1', 'longdup_20_r1', 'longdup_21_r1', 'longdup_22_r1', 'longdup_23_r1', 'longdup_24_r1', 'longdup_25_r1', 'longdup_7_r2', 'longdup_8_r2', 'longdup_9_r2', 'longdup_10_r2', 'longdup_11_r2', 'longdup_12_r2', 'longdup_13_r2', 'longdup_14_r2', 'longdup_15_r2', 'longdup_16_r2', 'longdup_17_r2', 'longdup_18_r2', 'longdup_19_r2', 'longdup_20_r2', 'longdup_21_r2', 'longdup_22_r2', 'longdup_23_r2', 'longdup_24_r2', 'longdup_25_r2', 'longdup_7_r3', 'longdup_8_r3', 'longdup_9_r3', 'longdup_10_r3', 'longdup_11_r3', 'longdup_12_r3', 'longdup_13_r3', 'longdup_14_r3', 'longdup_15_r3', 'longdup_16_r3', 'longdup_17_r3', 'longdup_18_r3', 'longdup_19_r3', 'longdup_20_r3', 'longdup_21_r3', 'longdup_22_r3', 'longdup_23_r3', 'longdup_24_r3', 'longdup_25_r3'],
  'clin': [str(s).strip() for s in open(_config.DATA_DIR + 'names_dislib.txt').readlines()[408:]]}

L3 = dict()
with open(_config.DATA_DIR + 'L3DataTable.csv') as f:
  for i, line in enumerate(f):
    w = line.strip().split(',')
    L3[w[0]] = [s for s in w[1:] if len(s) > 0]


##
# Library loading
##
def load_library(library_nm):
  lib_dir = _config.DATA_DIR + 'Libraries/'
  return pd.read_csv(lib_dir + '%s.csv' % (library_nm), dtype = {'Local Name': str, 'Name': str, 'Designed Name': str})

##
# Merging level 2 sets, for level 3 sets
##
def merge_l2_datasets_summation(l3_data):
  # Merges several l2 datasets using summation.
  # Preserves metadata and original count columns from each l2 dataseet.
  new_l3_data = dict()

  keys_superset = set()
  for k in l3_data:
    for key in l3_data[k]:
      if key not in keys_superset:
        keys_superset.add(key)

  print 'Merging %s level2 datasets with summation' % (len(l3_data))
  timer = util.Timer(total = len(keys_superset))
  for exp in keys_superset:
    dfs = []
    for l2_dataset in l3_data:
      if exp in l3_data[l2_dataset]:
        df = l3_data[l2_dataset][exp]
        dfs.append(df)

    mdf = dfs[0]
    # Label metadata for first dataframe
    metadata_cols = [s for s in mdf.columns if s[0] == '_']
    curr_l2_dataset_nm = l3_data.keys()[0]
    for meta_col in metadata_cols:
      mdf.rename(columns = {meta_col: meta_col + '_%s' % (curr_l2_dataset_nm)})
    mdf['Count_%s' % (curr_l2_dataset_nm)] = mdf['Count']
    count_cols = ['Count_%s' % (curr_l2_dataset_nm)]

    # Ensure sequence context and cutsite are all the same, otherwise merging is meaningless
    assert len(set(mdf['_Sequence Context'])) == 1
    assert len(set(mdf['_Cutsite'])) == 1
    expected_context = set(mdf['_Sequence Context']).pop()
    expected_cutsite = set(mdf['_Cutsite']).pop()

    # Merge dataframes
    join_cols = ['Category', 'Genotype Position', 'Indel with Mismatches', 'Ins Fivehomopolymer', 'Ins Template Length', 'Ins Template Orientation', 'Ins mh2', 'Ins p2', 'Inserted Bases', 'Length', 'Microhomology-Based']
    for idx in range(1, len(dfs)):
      curr_l2_dataset_nm = l3_data.keys()[idx]
      count_cols.append('Count_%s' % (curr_l2_dataset_nm))
      
      # Ensure sequence context and cutsite are all the same,   otherwise merging is meaningless
      assert len(set(dfs[idx]['_Sequence Context'])) == 1
      assert len(set(dfs[idx]['_Cutsite'])) == 1
      curr_context = set(dfs[idx]['_Sequence Context']).pop()
      curr_cutsite = set(dfs[idx]['_Cutsite']).pop()
      try:
        assert bool(curr_context == expected_context) and bool(curr_cutsite == expected_cutsite), 'ERROR: Trying to merge two L2 datasets with different context or cutsite'
      except:
        import code; code.interact(local=dict(globals(), **locals()))

      # Label metadata for more dataframes
      mdf = mdf.merge(dfs[idx], how = 'outer', on = join_cols, suffixes = ('', '_%s' % (curr_l2_dataset_nm)))

    # Clean up merged dataframe

    # Replace NaN in joined count columns with 0
    for count_col in count_cols:
      mdf[count_col].fillna(value = 0, inplace = True)

    # Use summation to merge counts
    mdf['Count'] = 0
    for count_col in count_cols:
      mdf['Count'] += mdf[count_col]

    # Standardize _sequence context and _cutsite columns
    mdf['_Sequence Context'] = expected_context
    mdf['_Cutsite'] = expected_cutsite

    # NaNs can exist in l2-specific metadata columns
    # NaNs can also exist in _ExpDir and _Experiment
    # But we guarantee that no NaNs exist in _Count (summed)
    # in _Sequence Context (shared sequence context)
    # and in _Cutsite (shared cutsite)

    # Store merged dataframe, key is level 1 name
    new_l3_data[exp] = mdf
    timer.update()
  return new_l3_data

##
# External facing: Load dataset regardless of level
##
def load_dataset(nm, exp_subset = None, exp_subset_col = None):
  if nm in set(D['Name']):
    return load_l2_dataset(nm, exp_subset = exp_subset, exp_subset_col = exp_subset_col)
  elif nm in L3:
    return load_l3_dataset(nm, exp_subset = exp_subset, exp_subset_col = exp_subset_col)
  else:
    print 'ERROR: Bad name %s' % (nm)
    return None

##
# Data Loading: Level 3 sets
##
def load_l3_dataset(nm, exp_subset = None, exp_subset_col = None):
  # Loads a level 3 dataset by loading its constitutive level 2 datasets and all of their single experiments.
  # Combines the level 2 datasets in some fashion (e.g., summation)
  # Pickles for amortized processing
  if nm not in L3:
    print 'ERROR: Bad L3 name %s' % (nm)
    return None
  print 'Loading L3 dataset: %s' % (nm)

  # Load library -- should just be one for all of L3 set
  l2_dataset_nm = L3[nm][0]
  row = D[D['Name'] == l2_dataset_nm]
  library_nm = row['Library'].iloc[0]
  library_df = load_library(library_nm)

  ## Prepare to filter and rename default dataset
  # Set which name column is used to label single experiments. By default, use 'Name' column
  nm_col = 'Name'
  if exp_subset_col is not None:
    nm_col = exp_subset_col
  if nm_col not in list(library_df.columns):
    print 'ERROR: Bad column name, %s' % (nm_col)

  # Determine subset of experiments to return. By default, use all
  exp_set = set(library_df[nm_col])
  if exp_subset is not None:
    if exp_subset not in exp_subset_dict:
      print 'ERROR: Bad exp subset %s' % (exp_subset)
    exp_set = exp_subset_dict[exp_subset]

  # Try to load from pickle
  pickled_fn = _config.DATA_DIR + 'L3_Datasets/%s.pkl' % (nm)
  if os.path.isfile(pickled_fn):
    print 'Loading dataset from pickle...'
    l3_data = pickle.load(open(pickled_fn))
    print 'Done'
  else:
    print 'Loading dataset from scratch...'
    l3_data = dict()
    for l2_dataset_nm in L3[nm]:
      data = load_l2_dataset(l2_dataset_nm, exp_subset = exp_subset, exp_subset_col = exp_subset_col)
      l3_data[l2_dataset_nm] = data
    l3_data = merge_l2_datasets_summation(l3_data)
    pickle.dump(l3_data, open(pickled_fn, 'w'))
    print 'Done'

  # Iterate through library, using local name to load dataframes, and labeling it with nm_col -- only on the subset of data specified
  dataset = dict()
  for idx, row2 in library_df.iterrows():
    if row2[nm_col] in exp_set:
      default_nm = row2['Name']
      d = l3_data[default_nm]
      dataset[row2[nm_col]] = d
  return dataset

##
# Data Loading: Level 2 sets
##
def load_l2_dataset(nm, exp_subset = None, exp_subset_col = None):
  # Loads a level 2 dataset by finding its associated library, and iteratively loading all single experiments in the library.
  # Stores the the level 2 dataset with pickle. 
  if nm not in set(D['Name']):
    print 'ERROR: Bad name %s' % (nm)
    return None
  print 'Loading %s' % (nm)

  row = D[D['Name'] == nm]
  library_nm = row['Library'].iloc[0]
  c6_dir = row['Alignment_Directory'].iloc[0]
  e_dir = row['Data_Directory'].iloc[0]
  local_nm = row['Local_Name'].iloc[0]
  memoize = row['Memoization'].iloc[0]

  library_df = load_library(library_nm)

  ## Prepare to filter and rename default dataset
  # Set which name column is used to label single experiments. By default, use 'Name' column
  nm_col = 'Name'
  if exp_subset_col is not None:
    nm_col = exp_subset_col
  if nm_col not in list(library_df.columns):
    print 'ERROR: Bad column name, %s' % (nm_col)

  # Determine subset of experiments to return. By default, use all
  exp_set = set(library_df[nm_col])
  if exp_subset is not None:
    if exp_subset not in exp_subset_dict:
      print 'ERROR: Bad exp subset %s' % (exp_subset)
    exp_set = exp_subset_dict[exp_subset]

  # If pickled version of default dataset exists, load it. Otherwise, make it.
  pickled_fn = _config.DATA_DIR + 'L2_Datasets/%s' % (nm)
  if os.path.isfile(pickled_fn):
    print 'Loading default dataset from pickle...'
    default_dataset = pickle.load(open(pickled_fn))
    print 'Done'
  else:
    print 'Loading default dataset from scratch...'
    default_dataset = dict()
    timer = util.Timer(total = len(library_df))
    for idx, row2 in library_df.iterrows():
      d = load_experiment(local_nm, e_dir, row2['Local Name'], memoize)
      default_dataset[row2['Name']] = d
      timer.update()
    pickle.dump(default_dataset, open(pickled_fn, 'w'))
    print 'Done'


  # Iterate through library, using local name to load dataframes, and labeling it with nm_col -- only on the subset of data specified
  dataset = dict()
  for idx, row2 in library_df.iterrows():
    if row2[nm_col] in exp_set:
      default_nm = row2['Name']
      d = default_dataset[default_nm]
      dataset[row2[nm_col]] = d

  print 'Dataset size: %s' % (len(dataset))
  if len(dataset) == 0:
    print 'Ensure exp_subset is a list of strings'

  return dataset

##
# Data Loading: Level 1 sets / single experiments
##
def load_experiment(l2_set_nm, l2_set_dir, default_exp_nm, memoize):
  # Load single csv
  if memoize == 'no':
    d = pd.read_csv(l2_set_dir + default_exp_nm + '.csv', index_col = 0, dtype = csv_dtypes)

  # Subset from larger df, memoizing it
  elif memoize == 'yes':
    if l2_set_nm not in memoized:
      l2_df = l2_set_dir + l2_set_nm + '.csv'
      df = pd.read_csv(l2_df, index_col = 0, dtype = csv_dtypes)
      memoized[l2_set_nm] = df
    else:
      df = memoized[l2_set_nm]

    # assumption-free version
    d = df[df['_Experiment'] == default_exp_nm]
  return d