from __future__ import division
import _config, _predict
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
import re
import random
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
from mylib import compbio
import pandas as pd
from scipy.stats import entropy

# Default params
DEFAULT_INP_DIR = _config.OUT_PLACE + 'a_split/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
exon_dfs_out_dir = out_dir + 'exon_dfs/'
intron_dfs_out_dir = out_dir + 'intron_dfs/'

rate_model = None
rate_model_nm = None
bp_model = None
bp_model_nm = None
normalizer = None
normalizer_nm = None

##
# Insertion Modeling Init
##
def init_rate_bp_models():
  global rate_model
  global rate_model_nm
  global bp_model
  global bp_model_nm
  global normalizer
  global normalizer_nm

  model_dir = '/cluster/mshen/prj/mmej_figures/out/e5_ins_ratebpmodel/'

  rate_model_nm = 'rate_model_v2'
  bp_model_nm = 'bp_model_v2'
  normalizer_nm = 'Normalizer_v2'

  print 'Loading %s...\nLoading %s...' % (rate_model_nm, bp_model_nm)
  with open(model_dir + '%s.pkl' % (rate_model_nm)) as f:
    rate_model = pickle.load(f)
  with open(model_dir + '%s.pkl' % (bp_model_nm)) as f:
    bp_model = pickle.load(f)
  with open(model_dir + '%s.pkl' % (normalizer_nm)) as f:
    normalizer = pickle.load(f)
  return

##
# Text parsing
##
def parse_header(header):
  w = header.split('_')
  gene_kgid = w[0].replace('>', '')
  chrom = w[1]
  start = int(w[2]) - 30
  end = int(w[3]) + 30
  data_type = w[4]
  return gene_kgid, chrom, start, end

##
# IO
##
def maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force = False):
  if split == '0':
    line_threshold = 500
  else:
    line_threshold = 5000
  norm_condition = bool(bool(len(dd['Unique ID']) > line_threshold) and bool(len(dd_shuffled['Unique ID']) > line_threshold))

  if norm_condition or force:
    print 'Flushing, num. %s' % (num_flushed)
    df_out_fn = out_dir + '%s_%s_%s.csv' % (data_nm, split, num_flushed)
    df = pd.DataFrame(dd)
    df.to_csv(df_out_fn)

    df_out_fn = out_dir + '%s_%s_shuffled_%s.csv' % (data_nm, split, num_flushed)
    df = pd.DataFrame(dd_shuffled)
    df.to_csv(df_out_fn)

    num_flushed += 1
    dd = defaultdict(list)
    dd_shuffled = defaultdict(list)
  else:
    pass
  return dd, dd_shuffled, num_flushed

##
# Prediction
##
def find_cutsites_and_predict(inp_fn, data_nm, split):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions

  _predict.init_model(run_iter = 'aax', param_iter = 'aag')
  dd = defaultdict(list)
  dd_shuffled = defaultdict(list)

  if data_nm == 'exons':
    df_out_dir = exon_dfs_out_dir
  elif data_nm == 'introns':
    df_out_dir = intron_dfs_out_dir

  num_flushed = 0
  timer = util.Timer(total = util.line_count(inp_fn))
  with open(inp_fn) as f:
    for i, line in enumerate(f):
      if i % 2 == 0:
        header = line.strip()
      if i % 2 == 1:
        sequence = line.strip()

        if len(sequence) < 60:
          continue
        if len(sequence) > 500000:
          continue

        bulk_predict(header, sequence, dd, dd_shuffled, df_out_dir)
        dd, dd_shuffled, num_flushed = maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed)

      if (i - 1) % 50 == 0 and i > 1:
        print '%s pct, %s' % (i / 500, datetime.datetime.now())

      timer.update()

  maybe_flush(dd, dd_shuffled, data_nm, split, num_flushed, force = True)
  return

def get_indel_len_pred(pred_all_df):
  indel_len_pred = dict()

  # 1 bp insertions
  crit = (pred_all_df['Category'] == 'ins')
  indel_len_pred[1] = float(sum(pred_all_df[crit]['Predicted_Frequency']))

  # Deletions
  for del_len in range(1, 60):
    crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Length'] == del_len)
    freq = float(sum(pred_all_df[crit]['Predicted_Frequency']))
    dl_key = -1 * del_len
    indel_len_pred[dl_key] = freq

  # Frameshifts, insertion-orientation
  fs = {'+0': 0, '+1': 0, '+2': 0}
  for indel_len in indel_len_pred:
    fs_key = '+%s' % (indel_len % 3)
    fs[fs_key] += indel_len_pred[indel_len]
  return indel_len_pred, fs


##
def bulk_predict(header, sequence, dd, dd_shuffled, df_out_dir):
  # Input: A specific sequence
  # Find all Cas9 cutsites, gather metadata, and run inDelphi
  try:
    ans = parse_header(header)
    gene_kgid, chrom, start, end = ans
  except:
    return

  for idx in range(len(sequence)):
    seq = ''
    if sequence[idx : idx+2] == 'CC':
      cutsite = idx + 6
      seq = sequence[cutsite - 30 : cutsite + 30]
      seq = compbio.reverse_complement(seq)
      orientation = '-'
    if sequence[idx : idx+2] == 'GG':
      cutsite = idx - 4
      seq = sequence[cutsite - 30 : cutsite + 30]
      orientation = '+'
    if seq == '':
      continue
    if len(seq) != 60:
      continue

    # Sanitize input
    seq = seq.upper()
    if 'N' in seq:
      continue
    if not re.match('^[ACGT]*$', seq):
      continue

    # Randomly query subset for broad shallow coverage
    r = np.random.random()
    if r > 0.05:
      continue

    # Shuffle everything but GG
    seq_nogg = list(seq[:34] + seq[36:])
    random.shuffle(seq_nogg)
    shuffled_seq = ''.join(seq_nogg[:34]) + 'GG' + ''.join(seq_nogg[36:])

    for d, seq_context, shuffled_nm in zip([dd, dd_shuffled], 
                                           [seq, shuffled_seq],
                                           ['wt', 'shuffled']):
      #
      # Store metadata statistics
      #
      local_cutsite = 30
      grna = seq_context[13:33]
      cutsite_coord = start + idx
      unique_id = '%s_%s_hg38_%s_%s_%s' % (gene_kgid, grna, chrom, cutsite_coord, orientation)

      d['Sequence Context'].append(seq_context)
      d['Local Cutsite'].append(local_cutsite)
      d['Chromosome'].append(chrom)
      d['Cutsite Location'].append(cutsite_coord)
      d['Orientation'].append(orientation)
      d['Cas9 gRNA'].append(grna)
      d['Gene kgID'].append(gene_kgid)
      d['Unique ID'].append(unique_id)

      # Make predictions
      ans = _predict.predict_all(seq_context, local_cutsite, 
                                 rate_model, bp_model, normalizer)
      pred_del_df, pred_all_df, total_phi_score, ins_del_ratio = ans

      # Save predictions
      # del_df_out_fn = df_out_dir + '%s_%s_%s.csv' % (unique_id, 'dels', shuffled_nm)
      # pred_del_df.to_csv(del_df_out_fn)
      # all_df_out_fn = df_out_dir + '%s_%s_%s.csv' % (unique_id, 'all', shuffled_nm)
      # pred_all_df.to_csv(all_df_out_fn)

      ## Translate predictions to indel length frequencies
      indel_len_pred, fs = get_indel_len_pred(pred_all_df)

      #
      # Store prediction statistics
      #
      d['Total Phi Score'].append(total_phi_score)
      d['1ins/del Ratio'].append(ins_del_ratio)

      d['1ins Rate Model'].append(rate_model_nm)
      d['1ins bp Model'].append(bp_model_nm)
      d['1ins normalizer'].append(normalizer_nm)

      d['Frameshift +0'].append(fs['+0'])
      d['Frameshift +1'].append(fs['+1'])
      d['Frameshift +2'].append(fs['+2'])
      d['Frameshift'].append(fs['+1'] + fs['+2'])

      crit = (pred_del_df['Genotype Position'] != 'e')
      s = pred_del_df[crit]['Predicted_Frequency']
      s = np.array(s) / sum(s)
      del_gt_precision = 1 - entropy(s) / np.log(len(s))
      d['Precision - Del Genotype'].append(del_gt_precision)
      
      dls = []
      for del_len in range(1, 60):
        dlkey = -1 * del_len
        dls.append(indel_len_pred[dlkey])
      dls = np.array(dls) / sum(dls)
      del_len_precision = 1 - entropy(dls) / np.log(len(dls))
      d['Precision - Del Length'].append(del_len_precision)
      
      crit = (pred_all_df['Genotype Position'] != 'e')
      s = pred_all_df[crit]['Predicted_Frequency']
      s = np.array(s) / sum(s)
      all_gt_precision = 1 - entropy(s) / np.log(len(s))
      d['Precision - All Genotype'].append(all_gt_precision)

      negthree_nt = seq_context[local_cutsite - 1]
      negfour_nt = seq_context[local_cutsite]
      d['-4 nt'].append(negfour_nt)
      d['-3 nt'].append(negthree_nt)

      crit = (pred_all_df['Category'] == 'ins')
      highest_ins_rate = max(pred_all_df[crit]['Predicted_Frequency'])
      crit = (pred_all_df['Category'] == 'del') & (pred_all_df['Genotype Position'] != 'e')
      highest_del_rate = max(pred_all_df[crit]['Predicted_Frequency'])
      d['Highest Ins Rate'].append(highest_ins_rate)
      d['Highest Del Rate'].append(highest_del_rate)

  return


##
# nohups
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print 'Generating nohup scripts...'
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  w_dir = _config.SRC_DIR
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  curr_num = 0
  num_scripts = 0
  nums = {'exons': 36, 'introns': 32}
  for typ in nums:
    for split in range(nums[typ]):
      script_id = NAME.split('_')[0]
      command = 'python -u %s.py %s %s' % (NAME, typ, split)

      script_abbrev = NAME.split('_')[0]
      sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_abbrev, typ, split)
      with open(sh_fn, 'w') as f:
        f.write('#!/bin/bash\n%s\n' % (command))
      curr_num += 1

      # Write qsub commands
      qsub_commands.append('qsub -m e -wd %s %s' % (_config.SRC_DIR, sh_fn))

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print 'Wrote %s shell scripts to %s' % (curr_num, qsubs_dir)
  return


##
# Main
##
@util.time_dec
def main(data_nm = '', split = ''):
  print NAME
  global out_dir
  util.ensure_dir_exists(out_dir)
  util.ensure_dir_exists(exon_dfs_out_dir)
  util.ensure_dir_exists(intron_dfs_out_dir)

  if data_nm == '' and split == '':
    gen_qsubs()
    return

  inp_fn = DEFAULT_INP_DIR + '%s_%s.fa' % (data_nm, split)
  init_rate_bp_models()
  find_cutsites_and_predict(inp_fn, data_nm, split)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(data_nm = sys.argv[1], split = sys.argv[2])
  else:
    main()
