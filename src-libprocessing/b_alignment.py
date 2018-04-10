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

# Default params
DEFAULT_INP_DIR = _config.OUT_PLACE + 'b3_splitunique/'
NAME = util.get_fn(__file__)

##
# gRNA and mismatch
##
def get_grna_idx(read):
  cand_idxs = None
  if read[:5] == 'GACGT':
    len_bc = len('GACGTcgatGGAAAGGACGAAACACCG')
  if read[:5] == 'GTGAC':
    len_bc = len('GTGACGGAAAGGACGAAACACCG')
  if read[:5] == 'AACGG':
    len_bc = len('AACGGGAAAGGACGAAACACCG')
  if read[:5] == 'CAGGG':
    len_bc = len('CAGGGAAAGGACGAAACACCG')
  grna = read[len_bc : len_bc + 20]
  if grna[0] == 'G':
    grna = read[len_bc - 1 : len_bc + 19]
  if grna in _config.d.GRNAS:
    return [_config.d.GRNAS.index(grna)]
  else:
    cand_idxs = []
    for idx, s in enumerate(_config.d.GRNAS):
      if dist(grna, s) == 1:
        cand_idxs.append(idx)
    return cand_idxs

def dist(grna, ref):
  return sum([bool(c1!=c2) for c1,c2 in zip(grna, ref)])

##
# Alignments
##
def alignment(read, cand_idxs):
  seq_align_tool = '/cluster/mshen/tools/seq-align/bin/needleman_wunsch'
  targets = [_config.d.TARGETS[i] for i in cand_idxs]
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
  for exp, target in enumerate(_config.d.TARGETS):
    kmers = get_lsh_kmers(target)
    for kmer in kmers:
      lsh_dict[kmer].append(exp)
  return lsh_dict

def get_lsh_kmers(target):
  kmer_len = 7
  kmers = []
  for idx in range(len(target) - kmer_len):
    kmer = target[idx : idx + kmer_len]
    kmers.append(kmer)
  return kmers

def find_best_designed_target(read, cand_idxs, lsh_dict):
  new_cand_idxs = copy.copy(cand_idxs)
  kmers = get_lsh_kmers(read)
  scores = dict()
  for kmer in kmers:
    for exp in lsh_dict[kmer]:
      if exp not in scores:
        scores[exp] = 0
      scores[exp] += 1
  best_exp = sorted(scores, key = scores.get)[-1]
  best_score = max(scores.values())

  cand_scores = [-5]
  for cand in cand_idxs:
    if cand in scores:
      cand_scores.append(scores[cand])
  best_cand_score = max(cand_scores)
  
  if best_cand_score + 5 >= best_score:
    # If found gRNA matches well, then don't add more candidates. This helps with similar designed targets.
    pass
  else:
    new_cand_idxs.append(best_exp)
  return new_cand_idxs

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
  for exp in range(2000):
    out_fn = out_dir + '%s.txt' % (exp)
    util.exists_empty_fn(out_fn)
  return

##
# Main
##
def matchmaker(inp_dir, out_dir, nm, split):
  print nm, split
  stdout_fn = _config.SRC_DIR + 'nh_c_%s_%s.out' % (nm, split)
  util.exists_empty_fn(stdout_fn)
  out_dir = out_dir + nm + '/' + split + '/'
  util.ensure_dir_exists(out_dir)
  inp_fn = inp_dir + nm + '/' + split + '.fa'

  count = 0
  num_recombined = 0
  num_unmatched_grnas = 0
  total = 0

  lsh_dict = build_targets_better_lsh()
  alignment_buffer = init_alignment_buffer()

  prepare_outfns(out_dir)

  tot_reads = util.line_count(inp_fn)
  timer = util.Timer(total = tot_reads)
  with open(inp_fn) as f:
    for i, line in enumerate(f):
      if i % 2 == 0:
        count = int(line.replace('>', ''))
      if i % 2 == 1:
        [l1, l2] = line.split()
        l2 = compbio.reverse_complement(l2)
        align_header = '>%s' % (count)

        # Try to find designed target from gRNA
        gRNA_cand_idxs = get_grna_idx(l1)
        if len(gRNA_cand_idxs) == 0:
          num_unmatched_grnas += count
          align_header += '_matchedgRNA-false'
        else:
          align_header += '_matchedgRNA-true'

        # Try to find designed target from LSH
        cand_idxs = find_best_designed_target(l2[-55:], gRNA_cand_idxs, lsh_dict)

        # Run alignment
        best_idx, align = alignment(l2.strip(), cand_idxs)

        # Determine if recombination occurred -- did LSH candidate get used?
        if best_idx not in gRNA_cand_idxs and len(gRNA_cand_idxs) > 0:
          num_recombined += count
          align_header += '_recombined-true'
        else:
          align_header += '_recombined-false'
        total += count

        # Store alignment into buffer
        store_alignment(alignment_buffer, best_idx, align_header, align)

        if i % int(tot_reads / 100) == 1 and i > 1:
          # Flush alignment buffer
          flush_alignments(alignment_buffer, out_dir)

          # Stats for the curious
          with open(stdout_fn, 'a') as outf:
            outf.write('Time: %s\n' % (datetime.datetime.now()))
            outf.write('Progress: %s\n' % (i / int(tot_reads / 100)) )
            outf.write('Matched gRNA rate / total barcoded reads: %s\n' % (1 - num_unmatched_grnas/total) )
            outf.write('Recombination rate / matched gRNAs: %s\n' % (num_recombined / (total - num_unmatched_grnas)))
      timer.update()
  
  # Final flush
  flush_alignments(alignment_buffer, out_dir)

  return


@util.time_dec
def main(inp_dir, out_dir, nm = '', split = ''):
  print NAME  
  util.ensure_dir_exists(out_dir)

  # Function calls
  matchmaker(inp_dir, out_dir, nm, split) 
  return out_dir


if __name__ == '__main__':
  if len(sys.argv) == 3:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/', nm = sys.argv[1], split = sys.argv[2])
  else:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/', nm = 'GH', split = '0')
