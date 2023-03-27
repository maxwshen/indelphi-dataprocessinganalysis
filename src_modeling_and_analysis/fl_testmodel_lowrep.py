from __future__ import division
import _config, _lib, _data, _predict, _predict2
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
from scipy.stats import pearsonr

# Default params
DEFAULT_INP_DIR = None
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
redo = False

##
# Run statistics
##
def calc_statistics(orig_df1, orig_df2, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  df1 = _lib.mh_del_subset(orig_df1)
  df1 = _lib.indels_without_mismatches_subset(df1)
  df2 = _lib.mh_del_subset(orig_df2)
  df2 = _lib.indels_without_mismatches_subset(df2)

  if bool(sum(df1['Count']) <= 1000) or bool(sum(df2['Count']) <= 1000):
    return
  df1['Frequency'] = _lib.normalize_frequency(df1)
  df2['Frequency'] = _lib.normalize_frequency(df2)

  join_cols = ['Category', 'Genotype Position', 'Length']
  mdf = df1.merge(df2, how = 'outer', on = join_cols, suffixes = ['_1', '_2'])
  mdf['Frequency_1'].fillna(value = 0, inplace = True)
  mdf['Frequency_2'].fillna(value = 0, inplace = True)
  
  r = pearsonr(mdf['Frequency_1'], mdf['Frequency_2'])[0]

  print exp, r

  alldf_dict['_Experiment'].append(exp)

  return alldf_dict

def prepare_statistics():
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  data_nm = '1207-mESC-Dislib-Cas9-Tol2-Biorep1-r1-controladj'
  # data_nm = '0105-mESC-Lib1-Cas9-Tol2-BioRep2-r1-controladj'
  dataset1 = _data.load_dataset(data_nm)

  data_nm = '1207-mESC-Dislib-Cas9-Tol2-Biorep1-r2-controladj'
  # data_nm = '0105-mESC-Lib1-Cas9-Tol2-BioRep3-r1-controladj'
  dataset2 = _data.load_dataset(data_nm)

  for exp in dataset1.keys():
    df1 = dataset1[exp]
    df2 = dataset2[exp]
    calc_statistics(df1, df2, exp, alldf_dict)

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  alldf = pd.DataFrame(alldf_dict)
  return alldf



##
# Main
##
@util.time_dec
def main(data_nm = '', redo_flag = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)

  if redo_flag == 'redo':
    global redo
    redo = True

  prepare_statistics()

  return


if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(data_nm = sys.argv[1])
  elif len(sys.argv) == 3:
    main(data_nm = sys.argv[1], redo_flag = sys.argv[2])
  else:
    main()
