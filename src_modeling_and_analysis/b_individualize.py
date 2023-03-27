from __future__ import division
import _config
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
DEFAULT_INP_DIR = _config.OUT_PLACE + 'a_gather/'
NAME = util.get_fn(__file__)

##
# Individualize
##
def individualize(inp_dir, out_dir):
  # a_gather produces large dataframes of 2000 experiments concatenated together.
  # extracting dataframes for each individual experiment is slow, while it's faster to just read in individual csv's for each experiment. (This functions produces individual csv's).

  for inp_fn in os.listdir(inp_dir):
    if not fnmatch.fnmatch(inp_fn, '*csv'):
      continue

    # if inp_fn not in ['PRL-Lib1-mES.csv', 'PRL-DisLib-mES.csv', 'Lib1-mES.csv']:
      # continue

    inp_nm = inp_fn.replace('.csv', '')
    out_fold = out_dir + inp_nm + '/'
    util.ensure_dir_exists(out_fold)

    df = pd.read_csv(inp_dir + inp_fn)
    exps = set(df['Experiment'])
    print inp_nm
    timer = util.Timer(total = len(exps))
    for exp in exps:
      out_fn = out_fold + '%s.csv' % (exp)
      d = df[df['Experiment'] == exp]
      d.to_csv(out_fn)
      timer.update()

  return

##
# Main
##
@util.time_dec
def main(inp_dir, out_dir):
  print NAME  
  util.ensure_dir_exists(out_dir)

  individualize(inp_dir, out_dir)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')
  else:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')
