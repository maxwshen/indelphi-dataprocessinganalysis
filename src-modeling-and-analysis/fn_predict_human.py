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

# Default params
DEFAULT_INP_DIR = '/cluster/mshen/prj/mmej_manda2/out/2017-10-27/mb_grab_exons/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'

##
# IO
##
def init_df_buffer():
  return pd.DataFrame()

def flush_df_buffer(df_buffer, df_buffer_nm):
  print 'Flushing %s...' % (df_buffer_nm)
  df_buffer.to_csv(out_dir + df_buffer_nm + '.csv')
  return

##
# Run statistics
##
def predict(inp_fn):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  _predict2.init_model(run_iter = 'aay', param_iter = 'aae')
  df_buffer = init_df_buffer()
  df_buffer_nm = ''

  timer = util.Timer(total = util.line_count(inp_fn))
  with open(inp_fn) as f:
    for i, line in enumerate(f):
      if i % 2 == 0:
        header = line.strip()
        if df_buffer_nm == '':
          df_buffer_nm = header

      if i % 2 == 1:
        sequence = line.strip()
        if len(sequence) < 60:
          continue
        df_buffer = add_del_profiles(header, sequence, df_buffer)

        print len(df_buffer)
        if len(df_buffer) > 100000:
          flush_df_buffer(df_buffer, df_buffer_nm)
          df_buffer_nm = ''
          df_buffer = init_df_buffer()
      timer.update()
  return


def add_del_profiles(header, sequence, df_buffer):
  for idx in range(len(sequence)):
    seq = ''
    if sequence[idx : idx+2] == 'CC':
      cutsite = idx + 6
      seq = sequence[cutsite - 30 : cutsite + 30]
      seq = compbio.reverse_complement(seq)
    if sequence[idx : idx+2] == 'GG':
      cutsite = idx - 4
      seq = sequence[cutsite - 30 : cutsite + 30]

    if seq != '':
      if len(seq) != 60:
        continue
      local_cutsite = 30
      pred_df = _predict2.predict_mhdel(seq, local_cutsite)
      
      pred_df['header'] = header
      pred_df['seq'] = sequence
      pred_df['pam'] = sequence[idx : idx + 2]
      pred_df['cutsite'] = cutsite
      pred_df['shuffled'] = 'no'
      df_buffer = df_buffer.append(pred_df, ignore_index = True)

      pre, post = list(seq[:34]), list(seq[36:])
      random.shuffle(pre)
      random.shuffle(post)
      shuffled_seq = ''.join(pre) + 'GG' + ''.join(post)
      shuffled_pred_df = _predict2.predict_mhdel(seq, local_cutsite)

      shuffled_pred_df['header'] = header
      shuffled_pred_df['seq'] = sequence
      shuffled_pred_df['pam'] = sequence[idx : idx + 2]
      shuffled_pred_df['cutsite'] = cutsite
      shuffled_pred_df['shuffled'] = 'yes'
      df_buffer = df_buffer.append(shuffled_pred_df, ignore_index = True)
  return df_buffer


##
# nohups
##
def gen_nohups():
  # Generate qsub shell scripts and commands for easy parallelization
  print 'Generating nohup scripts...'
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  nh_commands = []

  num_scripts = 0
  for data_nm in ['hg38_exons', 'hg38_exons_1pct', 'hg38_exons_20pct', 'hg38_introns', 'hg38_introns_1pct', 'hg38_introns_20pct']:
    script_id = NAME.split('_')[0]
    command = 'nohup python -u %s.py %s > nh_%s_%s.out &' % (NAME, data_nm, script_id, data_nm)
    nh_commands.append(command)

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(nh_commands))

  return


##
# Main
##
@util.time_dec
def main(data_nm = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)

  if data_nm == '':
    gen_nohups()
    return

  inp_fn = DEFAULT_INP_DIR + data_nm + '.fa'

  predict(inp_fn)

  return


if __name__ == '__main__':
  if len(sys.argv) == 2:
    main(data_nm = sys.argv[1])
  else:
    main()
