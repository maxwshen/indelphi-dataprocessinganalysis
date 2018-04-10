from __future__ import division
import _config
import sys, os, datetime, subprocess, math, pickle, imp
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
DEFAULT_INP_DIR = None
NAME = util.get_fn(__file__)


##
# Plotting
##
def gather(inp_dir, out_dir):

  # Gather hek lib1
  pf = '/cluster/mshen/prj/'
  csv_fns = [pf + 'mmej_manda2/out/2017-10-27/e10_control_adjustment/2k-HEK293T-Cas9-Tol2.csv', 
    pf + 'mmej_disease_mES/out/e10_control_adjustment/GH.csv', 
    pf + 'mmej_disease_mES/out/e10_control_adjustment/IJ.csv', 
    pf + 'mmej_disease_human/out/e10_control_adjustment/HEK2Lib-new-2000x.csv', 
    pf + 'mmej_disease_human/out/e10_control_adjustment/HEK2Lib-new-3000.csv', 
    pf + 'mmej_disease_human/out/e10_control_adjustment/HEK2Lib-new-3000to1500x.csv', 
    pf + 'mmej_disease_human/out/e10_control_adjustment/HEK2Lib-new-3000to2000x.csv', 
    pf + 'mmej_disease_human/out/e10_control_adjustment/HEK2Lib-old-1500x.csv', 
    pf + 'mmej_disease_human/out/e10_control_adjustment/HEK2Lib-old-2000x.csv', 
    pf + 'mmej_disease_human/out/e10_control_adjustment/HEK2Lib-old-2500x.csv']
  all_df = pd.DataFrame()
  for csv_fn in csv_fns:
    df = pd.read_csv(csv_fn)
    df = df.iloc[:, ~df.columns.duplicated()]
    all_df = all_df.iloc[:, ~all_df.columns.duplicated()]
    all_df = all_df.append(df, ignore_index = True)
  all_df.to_csv(out_dir + 'Lib1-HEK293T.csv')
  

  # Gather hct lib 1
  pf = '/cluster/mshen/prj/'
  csv_fns = [pf + 'mmej_manda2/out/2017-10-27/e10_control_adjustment/2k-HCT116-Cas9-Tol2.csv', 
    pf + 'mmej_disease_human/out/e10_control_adjustment/HCT2kLib-new-1000x.csv',
    pf + 'mmej_disease_human/out/e10_control_adjustment/HCT2kLib-new-1500x.csv',
    pf + 'mmej_disease_human/out/e10_control_adjustment/HCT2k-old-1000x.csv',
    pf + 'mmej_disease_human/out/e10_control_adjustment/HCT2k-old-2000x.csv']
  all_df = pd.DataFrame()
  for csv_fn in csv_fns:
    df = pd.read_csv(csv_fn)
    df = df.iloc[:, ~df.columns.duplicated()]
    all_df = all_df.iloc[:, ~all_df.columns.duplicated()]
    all_df = all_df.append(df, ignore_index = True)
  all_df.to_csv(out_dir + 'Lib1-HCT116.csv')

  # Gather mES lib 1
  pf = '/cluster/mshen/prj/'
  csv_fns = [pf + 'mmej_manda2/out/2017-10-27/e10_control_adjustment/2k-mES-Cas9-Tol2.csv',
    pf + 'mmej_manda3/subprjs/lib1_mESC/out/e10_control_adjustment/GH.csv',
    pf + 'mmej_manda3/subprjs/lib1_mESC/out/e10_control_adjustment/IJ.csv']
  all_df = pd.DataFrame()
  for csv_fn in csv_fns:
    df = pd.read_csv(csv_fn)
    df = df.iloc[:, ~df.columns.duplicated()]
    all_df = all_df.iloc[:, ~all_df.columns.duplicated()]
    all_df = all_df.append(df, ignore_index = True)
  all_df.to_csv(out_dir + 'Lib1-mES.csv')

  # Gather mES dislib
  pf = '/cluster/mshen/prj/'
  csv_fns = [pf + 'mmej_disease_mES/out/e10_control_adjustment/BC.csv',
    pf + 'mmej_disease_mES/out/e10_control_adjustment/DE.csv']
  all_df = pd.DataFrame()
  for csv_fn in csv_fns:
    df = pd.read_csv(csv_fn)
    df = df.iloc[:, ~df.columns.duplicated()]
    all_df = all_df.iloc[:, ~all_df.columns.duplicated()]
    all_df = all_df.append(df, ignore_index = True)
  all_df.to_csv(out_dir + 'DisLib-mES.csv')

  # Gather HEK dislib
  pf = '/cluster/mshen/prj/'
  csv_fns = [pf + 'mmej_disease_human/out/e_newgenotype/HEK-DisLib-2500to2500x.csv',
    pf + 'mmej_manda3/subprjs/dislib/out/e_newgenotype/AB.csv',
    pf + 'mmej_manda3/subprjs/dislib/out/e_newgenotype/CD.csv']
  all_df = pd.DataFrame()
  for csv_fn in csv_fns:
    df = pd.read_csv(csv_fn)
    df = df.iloc[:, ~df.columns.duplicated()]
    all_df = all_df.iloc[:, ~all_df.columns.duplicated()]
    all_df = all_df.append(df, ignore_index = True)
  all_df.to_csv(out_dir + 'DisLib-HEK293T.csv')

  # Gather U2OS dislib
  pf = '/cluster/mshen/prj/'
  csv_fns = [pf + 'mmej_disease_human/out/e10_control_adjustment/U2OS-DisLib-rep2.csv',
    pf + 'mmej_manda3/subprjs/dislib/out/e10_control_adjustment/EF.csv']
  all_df = pd.DataFrame()
  for csv_fn in csv_fns:
    df = pd.read_csv(csv_fn)
    df = df.iloc[:, ~df.columns.duplicated()]
    all_df = all_df.iloc[:, ~all_df.columns.duplicated()]
    all_df = all_df.append(df, ignore_index = True)
  all_df.to_csv(out_dir + 'DisLib-U2OS.csv')

  # Gather U2OS/HEK dislib mix
  pf = '/cluster/mshen/prj/'
  csv_fns = [pf + 'mmej_disease_human/out/e10_control_adjustment/U2OS-DisLib-rep2.csv',
    pf + 'mmej_disease_human/out/e_newgenotype/HEK-DisLib-2500to2500x.csv',
    pf + 'mmej_disease_human/out/e_newgenotype/E-rev2.csv',
    pf + 'mmej_manda3/subprjs/dislib/out/e_newgenotype/AB.csv',
    pf + 'mmej_manda3/subprjs/dislib/out/e_newgenotype/CD.csv',
    pf + 'mmej_manda3/subprjs/dislib/out/e10_control_adjustment/EF.csv']
  all_df = pd.DataFrame()
  for csv_fn in csv_fns:
    df = pd.read_csv(csv_fn)
    if 'Adjusted Count' not in df.columns:
      df['Adjusted Count'] = df['Count']
    df = df.iloc[:, ~df.columns.duplicated()]
    all_df = all_df.iloc[:, ~all_df.columns.duplicated()]
    all_df = all_df.append(df, ignore_index = True)
  all_df.to_csv(out_dir + 'DisLib-U2OS-HEK-Mixture.csv')

  # Gather PRL lib1
  pf = '/cluster/mshen/prj/'
  csv_fns = [pf + 'mmej_PRL_022618/out/e_newgenotype/PRL_mES_Lib1_Cas9.csv']
  all_df = pd.DataFrame()
  for csv_fn in csv_fns:
    df = pd.read_csv(csv_fn)
    df = df.iloc[:, ~df.columns.duplicated()]
    all_df = all_df.iloc[:, ~all_df.columns.duplicated()]
    all_df = all_df.append(df, ignore_index = True)
  all_df.to_csv(out_dir + 'PRL-Lib1-mES.csv')

  # Gather PRL dislib
  pf = '/cluster/mshen/prj/'
  csv_fns = [pf + 'mmej_PRL_022618/out/e_newgenotype/PRL_mES_DisLib_Cas9.csv']
  all_df = pd.DataFrame()
  for csv_fn in csv_fns:
    df = pd.read_csv(csv_fn)
    df = df.iloc[:, ~df.columns.duplicated()]
    all_df = all_df.iloc[:, ~all_df.columns.duplicated()]
    all_df = all_df.append(df, ignore_index = True)
  all_df.to_csv(out_dir + 'PRL-DisLib-mES.csv')

  return

##
# Main
##
@util.time_dec
def main(inp_dir, out_dir):
  print NAME  
  util.ensure_dir_exists(out_dir)

  gather(inp_dir, out_dir)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')
  else:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')
