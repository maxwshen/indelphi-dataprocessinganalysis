# Preloads data for amortized processing
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, math, pickle, imp
sys.path.append('/cluster/mshen/')
import fnmatch
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

import _data
NAME = util.get_fn(__file__)

##
# Preloading
##
def data_preload(nm):
  l2_data = _data.load_dataset(nm)
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

  l2_names = _data.D['Name']
  l3_names = _data.L3

  num_scripts = 0
  for nm in l2_names:
    command = 'python %s.py %s' % (NAME, nm)
    script_id = 'pre-l2'

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, nm)
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

@util.time_dec
def main(nm = ''):
  # Function calls
  if nm == '':
    gen_qsubs()
    return

  data_preload(nm)
  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(nm = sys.argv[1])
  else:
    main()
