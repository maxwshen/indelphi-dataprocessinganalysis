# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, imp
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
import seaborn as sns

# Default params
DEFAULT_INP_DIR = _config.DATA_DIR
NAME = util.get_fn(__file__)


##
# 2bit
##
def query_genome(genome_build, chrm, start, align_len, srr_id):
  tool = '/cluster/mshen/tools/2bit/twoBitToFa'
  twobit_db = '/cluster/mshen/tools/2bit/%s.2bit' % (genome_build)
  start -= 1
  end = start + align_len
  command = '%s -seq=%s -start=%s -end=%s %s temp_%s.fa; cat temp_%s.fa' % (tool, chrm, start, end, twobit_db, srr_id, srr_id)
  query = subprocess.check_output(command, shell = True)
  genome = ''.join(query.split()[1:])
  return genome

def get_align_len(read, cigar):
  # Given a read and cigar string, figure out how many bases to query from 2bit genome
  baselen = len(read)
  cigs = parse_cigar(cigar)
  for element in cigs:
    typ = element[-1]
    num = int(element[:-1])
    if typ in ['I', 'S']:
      baselen -= num
    if typ in ['D', 'N']:
      baselen += num
  return baselen


##
# Support
##
def get_expected_chrm_pos(srr_id):
  # ex: hg38_chr17:81511308-81511330
  T = _config.d.TABLE
  srr_row = T[T['Run'] == srr_id]
  cloc = str(srr_row['chromosome_loc']).split()[1]
  genome_build = str(cloc.split('_')[0])
  cloc = cloc.split('_')[1]
  exp_chrm = str(cloc.split(':')[0])
  exp_pos = int(cloc.split(':')[1].split('-')[0])
  return genome_build, exp_chrm, exp_pos

def parse_cigar(cigar):
  elements = []
  opset = set(['M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X'])
  trailing_idx = 0
  for idx, char in enumerate(cigar):
    if char in opset:
      element = cigar[trailing_idx : idx + 1]
      elements.append(element)
      trailing_idx = idx + 1
  return elements

##
# Construct align
##
def construct_align(read, genome, cigar, start):
  s1, s2 = '', ''
  idx1, idx2 = 0, 0
  cigs = parse_cigar(cigar)
  for element in cigs:
    typ = element[-1]
    num = int(element[:-1])
    if typ in ['M', '=', 'X']:
      s1 += read[idx1 : idx1 + num]
      s2 += genome[idx2 : idx2 + num]
      idx1 += num
      idx2 += num
    if typ in ['I', 'S']:
      s1 += read[idx1 : idx1 + num]
      s2 += '-' * num
      idx1 += num
    if typ in ['D', 'N']:
      s1 += '-' * num
      s2 += genome[idx2 : idx2 + num]
      idx2 += num
  return '%s\n%s\n%s\n\n' % (start, s1, s2)

##
# Main: Convert SAM to raw alignment
##
def convert_alignment(srr_id, out_dir):
  print srr_id
  if srr_id not in _config.d.RUNS_SET:
    return 'Bad srr_id %s' % (srr_id)
  sam_fn = _config.d.sam_fn(srr_id)
  genome_build, exp_chrm, exp_pos = get_expected_chrm_pos(srr_id)
  
  num_aligns, num_distant = 0, 0
  align_collection = defaultdict(lambda: 0)

  timer = util.Timer(total = util.line_count(sam_fn))
  with open(sam_fn) as f:
    for i, line in enumerate(f):
      if not line.startswith('@'):
        num_aligns += 1
        chrm = line.split()[2]
        start = int(line.split()[3])
        cigar = line.split()[5]
        read = line.split()[9]

        if abs(exp_pos - start) > 1000:
          num_distant += 1
          continue

        align_len = get_align_len(read, cigar)
        genome = query_genome(genome_build, chrm, start, align_len, srr_id)
        align = construct_align(read, genome, cigar, start)
        align_collection[align] += 1
      timer.update()

  sorted_aligns = sorted(align_collection, key = align_collection.get, reverse = True)

  out_fn = out_dir + '%s.txt' % (srr_id)
  with open(out_fn, 'w') as f:
    for align in sorted_aligns:
      count = align_collection[align]
      f.write('>%s_%s' % (count, align))

  print '%s distant out of %s alignments: %s' % (num_distant, num_aligns, num_distant / num_aligns)
  print 'Done'
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
  for idx in range(3696622, 3702820 + 1):
    command = 'python %s.py SRR%s' % (NAME, idx)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, str(idx)[-4:])
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -M maxwshen@csail.mit.edu -m e -wd %s %s' % (_config.SRC_DIR, sh_fn))

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print 'Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir)
  return

##
# Main
##
@util.time_dec
def main(inp_dir, out_dir, srr_id = None):
  print NAME  
  util.ensure_dir_exists(out_dir)

  # Function calls
  if srr_id is None:
    gen_qsubs()
  else:
    convert_alignment(srr_id, out_dir)

  return out_dir


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(DEFAULT_INP_DIR, 
         _config.OUT_PLACE + NAME + '/', 
         srr_id = sys.argv[1])
  else:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')