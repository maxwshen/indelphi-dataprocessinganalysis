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
inp_dir = _config.OUT_PLACE + 'e10_control_adjustment/'
NAME = util.get_fn(__file__)

@util.time_dec
def main():
  print NAME  
  dtypes = {'Category': str, 'Count': float, 'Genotype Position': float, 'Indel with Mismatches': str, 'Ins Fivehomopolymer': str, 'Ins Template Length': float, 'Ins mh2': str, 'Ins p2': str, 'Inserted Bases': str, 'Length': float, 'Microhomology-Based': str, '_ExpDir': str, '_Experiment': str, '_Sequence Context': str, '_Cutsite': int}

  mdf = pd.DataFrame()
  bcs = [
    # '051018_U2OS_+_LibA_preCas9', 
    '052218_U2OS_+_LibA_postCas9_rep2', 
    '052218_U2OS_+_LibA_postCas9_rep1'
  ]
  for nm in bcs:
    mdf = pd.DataFrame()
    for start in range(0, 2000, 100):
      print start
      fn = inp_dir + '%s_genotypes_%s.csv' % (nm, start)
      df = pd.read_csv(fn, index_col = 0, dtype = dtypes)
      mdf = pd.concat([mdf, df], ignore_index = True)

    mdf.to_csv(inp_dir + '%s.csv' % (nm))
    print 'Wrote to %s.csv' % (inp_dir + '%s' % (nm))
  print 'Done'
  return


if __name__ == '__main__':
  main()
