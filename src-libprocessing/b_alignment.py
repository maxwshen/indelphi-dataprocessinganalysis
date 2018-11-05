# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, copy
import numpy as np
from collections import defaultdict
sys.path.append('/cluster/mshen/')
from mylib import util
from mylib import compbio
from itertools import izip
import pickle
import pandas as pd

# Default params
inp_dir = _config.OUT_PLACE + 'a_split/'
NAME = util.get_fn(__file__)
out_place = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_place)
exp_design = pd.read_csv(_config.DATA_DIR + 'LibA.csv')

names_targets = dict()
for idx, row in exp_design.iterrows():
  names_targets[row['Name']] = row['Full target sequence']

##
# Alignments
##
def alignment(read, cand_idxs):
  seq_align_tool = '/cluster/mshen/tools/seq-align/bin/needleman_wunsch'
  targets = [names_targets[nm] for nm in cand_idxs]
  aligns = []
  for target_seq in targets:
    try:
      targetse = 'TCCGTGCTGTAACGAAAGGATGGGTGCGACGCGTCAT' + target_seq
      align = subprocess.check_output(seq_align_tool + ' --match 1 --mismatch -1 --gapopen -5 --gapextend -0 --freestartgap ' + read + ' ' + targetse, shell = True)
      aligns.append(align)
    except:
      pass

  if len(aligns) > 1:
    best_align = pick_best_alignment(aligns)
    best_idx = cand_idxs[aligns.index(best_align)]
  else:
    best_align = aligns[0]
    best_idx = cand_idxs[0]
  return best_idx, best_align

def pick_best_alignment(aligns):
  scores = []
  for align in aligns:
    w = align.split()
    s1, s2 = w[0], w[1]
    score = 0
    for i in range(len(s1)):
      if s1[i] == s2[i] and s1[i] != '-':
        score += 1
    scores.append(score)
  best_idx = scores.index(max(scores))
  return aligns[best_idx]

##
# Locality sensitive hashing
##
def build_targets_better_lsh():
  lsh_dict = defaultdict(list)
  for nm in names_targets:
    target = names_targets[nm]
    kmers = get_lsh_kmers(target)
    for kmer in kmers:
      lsh_dict[kmer].append(nm)
  return lsh_dict

def get_lsh_kmers(target):
  kmer_len = 7
  kmers = []
  for idx in range(len(target) - kmer_len):
    kmer = target[idx : idx + kmer_len]
    kmers.append(kmer)
  return kmers

def find_best_designed_target(read, lsh_dict):
  kmers = get_lsh_kmers(read)
  scores = dict()
  for kmer in kmers:
    for exp in lsh_dict[kmer]:
      if exp not in scores:
        scores[exp] = 0
      scores[exp] += 1

  if len(scores) == 0:
    return []

  sorted_scores = sorted(scores, key = scores.get, reverse = True)
  best_score = scores[sorted_scores[0]]
  # cand_idxs = []
  # for exp in sorted_scores:
  #   if scores[exp] + 5 < best_score:
  #     break
  #   cand_idxs.append(exp)
  cand_idxs = [sorted_scores[0]]
  return cand_idxs

##
# IO
##
def store_alignment(alignment_buffer, idx, align_header, align):
  align_string = '%s\n%s' % (align_header, align)
  alignment_buffer[idx].append(align_string)
  return

def init_alignment_buffer():
  alignment_buffer = defaultdict(list)
  return alignment_buffer

def flush_alignments(alignment_buffer, out_dir):
  print 'Flushing... \n%s' % (datetime.datetime.now())
  for exp in alignment_buffer:
    with open(out_dir + '%s.txt' % (exp), 'a') as f:
      for align in alignment_buffer[exp]:
        f.write(align)
  alignment_buffer = init_alignment_buffer()
  print 'Done flushing.\n%s' % (datetime.datetime.now())
  return

def prepare_outfns(out_dir):
  for exp in names_targets:
    out_fn = out_dir + '%s.txt' % (exp)
    util.exists_empty_fn(out_fn)
  return

##
# Main
##
def matchmaker(nm, split):
  print nm, split
  stdout_fn = _config.SRC_DIR + 'nh_c_%s_%s.out' % (nm, split)
  util.exists_empty_fn(stdout_fn)
  out_dir = out_place + nm + '/' + split + '/'
  util.ensure_dir_exists(out_dir)

  inp_fn = inp_dir + '%s_r2_%s.fq' % (nm, split)

  lsh_dict = build_targets_better_lsh()
  alignment_buffer = init_alignment_buffer()

  prepare_outfns(out_dir)

  qf = 0

  tot_reads = util.line_count(inp_fn)
  timer = util.Timer(total = tot_reads)
  from itertools import izip
  with open(inp_fn) as f:
    for i, line in enumerate(f):
      if i % 4 == 0:
        pass
      if i % 4 == 1:
        l2 = line.strip()
      if i % 4 == 3:
        # Quality filter
        q2 = line.strip()
        qs = [ord(s)-33 for s in q2]
        if np.mean(qs) < 28:
          qf += 1
          continue

        l2 = compbio.reverse_complement(l2)
        align_header = '>1'

        # Try to find designed target from LSH
        cand_idxs = find_best_designed_target(l2, lsh_dict)
        if len(cand_idxs) == 0:
          continue

        # Run alignment
        best_idx, align = alignment(l2, cand_idxs)

        # Store alignment into buffer
        store_alignment(alignment_buffer, best_idx, align_header, align)

      if i % int(tot_reads / 100) == 1 and i > 1:
        # Flush alignment buffer
        flush_alignments(alignment_buffer, out_dir)
        alignment_buffer = init_alignment_buffer()

        # Stats for the curious
        with open(stdout_fn, 'a') as outf:
          outf.write('Time: %s\n' % (datetime.datetime.now()))
          outf.write('Progress: %s\n' % (i / int(tot_reads / 100)) )
          outf.write('Quality filtered pct: %s\n' % (qf / (i/4)))
      timer.update()
  
  # Final flush
  flush_alignments(alignment_buffer, out_dir)

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

  num_scripts = 0
  for _nm in ['051018_U2OS_+_LibA_preCas9', '052218_U2OS_+_LibA_postCas9_rep1', '052218_U2OS_+_LibA_postCas9_rep2']:
    for _split in range(60):
      command = 'python %s.py %s %s' % (NAME, _nm, _split)
      script_id = NAME.split('_')[0]

      # Write shell scripts
      sh_fn = qsubs_dir + 'q_%s_%s_%s.sh' % (script_id, _nm, _split)
      with open(sh_fn, 'w') as f:
        f.write('#!/bin/bash\n%s\n' % (command))
      num_scripts += 1

      # Write qsub commands
      qsub_commands.append('qsub -m e -wd %s %s' % (_config.SRC_DIR, sh_fn))

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print 'Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir)
  return

@util.time_dec
def main(nm = '', split = ''):
  print NAME  

  if nm == '' and split == '':
    gen_qsubs()
    return

  # Function calls
  matchmaker(nm, split) 
  return


if __name__ == '__main__':
  if len(sys.argv) > 2:
    main(nm = sys.argv[1], split = sys.argv[2])
  else:
    main()
