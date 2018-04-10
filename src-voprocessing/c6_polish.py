# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
from mylib import compbio
from itertools import izip
import pickle

# Default params
DEFAULT_INP_DIR = _config.OUT_PLACE + 'a_convertsam/'
NAME = util.get_fn(__file__)

master_expected_cutsite = None  # 1 based
expected_cutsite = None
reverse_flag = False

##
# Cutsite
##
def set_master_expected_cutsite(srr_id):
  # Use designed table to determine genomic coordinates of designed cutsite. Query gRNA using twobit, determine gRNA orientation, then map cutsite from gRNA. Determines 1-based cutsite locus. 
  global master_expected_cutsite
  global reverse_flag
  T = _config.d.TABLE
  srr_row = T[T['Run'] == srr_id]
  cloc = str(srr_row['chromosome_loc']).split()[1]
  genome_build = str(cloc.split('_')[0])
  cloc = cloc.split('_')[1]
  chrm = str(cloc.split(':')[0])
  start = int(cloc.split(':')[1].split('-')[0])
  end = int(cloc.split(':')[1].split('-')[1])

  tool = '/cluster/mshen/tools/2bit/twoBitToFa'
  twobit_db = '/cluster/mshen/tools/2bit/%s.2bit' % (genome_build)
  twobit_start = start - 1  # convert 1-based from table to 0-based for twobit
  command = '%s -seq=%s -start=%s -end=%s %s temp_%s.fa; cat temp_%s.fa' % (tool, chrm, twobit_start, end, twobit_db, srr_id, srr_id)
  query = subprocess.check_output(command, shell = True)
  genome = ''.join(query.split()[1:]).upper()
  if genome[:2] == 'CC' and genome[-2:] != 'GG':
    master_expected_cutsite = start + 6
    reverse_flag = True
  elif genome[:2] != 'CC' and genome[-2:] == 'GG':
    master_expected_cutsite = start + 23 - 6
    reverse_flag = False
  elif genome[:2] == 'CC' and genome[-2:] == 'GG':
    # If both CC and GG are available, default to GG. 
    # Three out of 96 spacers have both CC/GG, all three are GG.
    master_expected_cutsite = start + 23 - 6
    reverse_flag = False
  else:
    print 'ERROR: Expected gRNA lacks NGG on both strands'
    sys.exit(0)
  print 'Reverse: %s' % (reverse_flag)
  return

##
# Left or right
##
def left_or_right_event(seq, cutsite):
  # Given a sequence with exactly one - event and possible end gaps, determine if it's on the left or right of the cutsite.
  left_test = len(seq[:cutsite + 1].replace('-', ' ').split())
  right_test = len(seq[cutsite - 1:].replace('-', ' ').split())
  if left_test == 2 and right_test == 1:
    return 'left'
  if left_test == 1 and right_test == 2:
    return 'right'

###
# IO
###
def prepare_align_outdirs(out_plc, start, end):
  util.ensure_dir_exists(out_plc)
  timer = util.Timer(total = end - start + 1)
  for exp in range(start, end + 1):
    out_idx_dir = out_plc + 'SRR' + str(exp) + '/'
    util.ensure_dir_exists(out_idx_dir)
    if len(os.listdir(out_idx_dir)) > 0:
      subprocess.check_output('rm -rf %s*' % (out_idx_dir), shell = True)
    timer.update()
  return

def save_alignments(data, out_dir):
  for repair_type in data:
    out_fn = out_dir + '%s.txt' % (repair_type)
    aligns = sort_combine_alignments(data[repair_type])
    sorted_keys = sorted(aligns, key = aligns.get, reverse = True)
    with open(out_fn, 'w') as f:
      for alignment in sorted_keys:
        total_count = aligns[alignment]
        f.write('>%s_%s' % (total_count, alignment))
  return

def sort_combine_alignments(aligns):
  store = dict()
  for idx, line in enumerate(aligns):
    if idx % 4 == 0:
      header = line
      count_section = header.split('_')[0]
      rest_section = '_'.join(header.split('_')[1:])
      count = int(count_section.replace('>', ''))
    if idx % 4 == 1:
      read = line
    if idx % 4 == 2:
      genome = line
      alignment = rest_section + '\n' + read + '\n' + genome + '\n\n'
      if alignment not in store:
        store[alignment] = 0
      store[alignment] += count
  return store


##
# Basic Sequence operations
##
def match_rate(s1, s2):
  assert len(s1) == len(s2)
  return sum([bool(c1==c2) for c1,c2 in zip(s1, s2)]) / len(s1)

def get_cutsite_idx(read, genome):
  # Given cutsite position in reference alone, 
  # find idx for this particular alignment such that 
  # read[:idx], read[idx:], &
  # genome[:idx], genome[idx:] 
  # are the CRISPR cut products...
  #   even with starting gaps and arbitrary indels
  genome_start_idx = genome.index(genome.replace('-', '')[:1])
  counter = 0
  for jdx in range(genome_start_idx, len(genome)):
    if counter == expected_cutsite:
      break
    if genome[jdx] in 'ACGTN':
      counter += 1
  cut_idx = jdx
  return cut_idx

def count_indels(s1, s2):
  num_dels, num_ins = 0, 0
  ts1, ts2 = trim_start_end_dashes(s1), trim_start_end_dashes(s2)
  num_dels = len(ts1.replace('-', ' ').split()) - 1
  num_ins = len(ts2.replace('-', ' ').split()) - 1
  return num_dels, num_ins

def trim_start_end_dashes(seq):
  alphabet = set(list(seq))
  if '-' in alphabet:
    alphabet.remove('-')
  alphabet = list(alphabet)
  start = min([seq.index(s) for s in alphabet])
  end = min([seq[::-1].index(s) for s in alphabet])
  if end > 0:
    return seq[start:-end]
  else:
    return seq[start:]

def findinf(seq, query):
  # if query not in seq, return large number. 
  # If query in seq, return its index.
  if query not in seq:
    return np.inf
  else:
    return seq.index(query)

##
# Measure alignment properties 
##
def score_single_deletion(read, genome):
  # Find where cut should be, with possible starting gaps, but exactly one deletion and no insertions in alignment.
  cutsite = get_cutsite_idx(read, genome)
  # seq[:cutsite], seq[cutsite:] gives cut products

  if '-' not in read[cutsite - 2 : cutsite + 2]:
    return ('del_notcrispr', 0)
  if read[cutsite - 1] != '-' and read[cutsite] != '-':
    return ('del_notatcut', 1)
  else:
    del_len = trim_start_end_dashes(read).count('-')
    return ('del%s' % (del_len), 2)

def score_single_insertion(read, genome):
  # Find where cut should be, with possible starting gaps, but exactly one insertion and no deletions in alignment.
  cutsite = get_cutsite_idx(read, genome)
  # seq[:cutsite], seq[cutsite:] gives cut products

  if '-' not in genome[cutsite - 2 : cutsite + 2]:
    # Indel in 10bp window requires looking at 12bp, just like how indel in 0bp window requires looking at 2bp
    return ('ins_notcrispr', 0)
  if genome[cutsite - 1] != '-' and genome[cutsite] != '-':
    return ('ins_notatcut', 1)
  else:
    ins_len = trim_start_end_dashes(genome).count('-')
    return ('ins%s' % (ins_len), 2)

def detect_wildtype(read, genome):
  num_dels, num_ins = count_indels(read, genome)
  if num_dels == 0 and num_ins == 0:
    if '-' not in trim_start_end_dashes(read) and '-' not in trim_start_end_dashes(genome):
      return True
  return False

def check_matches_threshold(read, genome):
  # In order to detect deletions up to 28bp as faithfully as possible, the 55bp designed context part of the alignment should have at least 20 bp matches.
  s1, s2 = '', ''
  for idx in range(len(genome) - 1, -1, -1):
    if genome[idx] in 'ACGTN':
      if read[idx] != '-':
        s1 += read[idx]
        s2 += genome[idx]
  if len(s1) == 0:
    return False
  match_score = match_rate(s1, s2)
  if len(s1) <= 5 or match_score <= 0.80:
    return False
  return True

def detect_pcr_recombination_anywhere(read, genome):
  # Alignment signature of PCR recombination:
  #   1. long insertion, chance overlap, long deletion
  #   2. long deletion, chance overlap, long insertion

  ir, ig = read + '*', genome + '*'
  ir, ig, element_nm, package = pcr_chomp_process_elements(ir, ig)

  match_threshold = 0.80
  elements = [element_nm]
  packages = [package]

  while element_nm != 'done':
    ir, ig, element_nm, package = pcr_chomp_process_elements(ir, ig)
    elements.append(element_nm)
    packages.append(package)

  # Check for signature with random matches
  for idx in range(2, len(elements)):
    check_recombination_trio = False
    if elements[idx-2] == 'ins' and elements[idx-1] == 'match' and elements[idx] == 'del':
      check_recombination_trio = True
      ins_len, del_len = packages[idx-2], packages[idx]
      
    if elements[idx-2] == 'del' and elements[idx-1] == 'match' and elements[idx] == 'ins':
      check_recombination_trio = True
      ins_len, del_len = packages[idx], packages[idx-2]
    
    if check_recombination_trio:
      s1, s2 = packages[idx-1]
      match_score = match_rate(s1, s2)
      match_len = len(s1)
      both_long = bool(ins_len >= 10 and del_len >= 10)
      one_verylong = bool(ins_len >= 30 or del_len >= 30)
      if both_long or one_verylong:
        if match_score <= match_threshold:
          return True
        if match_len <= 5:
          return True
        if len(set(s2)) == 1:
          return True
      if bool(ins_len >= 30 and del_len >= 30):
        if match_score <= 0.95:
          return True
        if match_len <= 10:
          return True
        if len(set(s2)) == 1:
          return True

  # Check for signature with 0 matches
  for idx in range(1, len(elements)):
    check_recombination_duo = False
    if elements[idx-1] == 'ins' and elements[idx] == 'del':
      check_recombination_duo = True
      ins_len, del_len = packages[idx-1], packages[idx]

    if elements[idx-1] == 'del' and elements[idx] == 'ins':
      check_recombination_duo = True
      ins_len, del_len = packages[idx], packages[idx-1]

    if check_recombination_duo:
      return True

  return False

def pcr_chomp_process_elements(read, genome):
  if read[0] == '*' or genome[0] == '*':
    return read, genome, 'done', None

  if read[0] == '-' and genome[0] != '-':
    element_nm = 'del'
    length = read.index(read.replace('-', '')[:1])
    package = length
  if read[0] != '-' and genome[0] == '-':
    element_nm = 'ins'
    length = genome.index(genome.replace('-', '')[:1])
    package = length
  if read[0] != '-' and genome[0] != '-':
    element_nm = 'match'
    read_idx = min(findinf(read, '-'), findinf(read, '*'))
    genome_idx = min(findinf(genome, '-'), findinf(genome, '*'))
    length = min(read_idx, genome_idx)
    package = (read[:length], genome[:length])

  new_read = read[length:]
  new_genome = genome[length:]
  return new_read, new_genome, element_nm, package

def detect_combination_indel(read, genome):
  # Counts indels in 10 bp window centered at cutsite
  cutsite = get_cutsite_idx(read, genome)
  num_dels, num_ins = count_indels(read, genome)
  total_indels = num_dels + num_ins

  left_idx = 0
  counter = 0
  for idx in range(cutsite -1, -1, -1):
    if genome[idx] != '-' and read[idx] != '-':
      counter += 1
    if counter == 5:
      break
  left_idx = idx

  right_idx = 0
  counter = 0
  for idx in range(cutsite, len(genome),):
    if counter == 5:
      break
    if genome[idx] != '-' and read[idx] != '-':
      counter += 1
  right_idx = idx

  # Regions between and surrounding indels should have match score at least 80% if they are real combination indels, otherwise they're probably recombination products -- label as other.  
  ir, ig = read + '*', genome + '*'
  ir, ig, element_nm, package = pcr_chomp_process_elements(ir, ig)
  elements = [element_nm]
  packages = [package]
  while element_nm != 'done':
    ir, ig, element_nm, package = pcr_chomp_process_elements(ir, ig)
    elements.append(element_nm)
    packages.append(package)

  for element, package in zip(elements, packages):
    if element == 'match':
      s1, s2 = package
      match_score = match_rate(s1, s2)
      if match_score <= 0.80:
        return 'other'

  # If match score test passes, 
  #   if 0 indels in 10bp window: label as forgiven wt (not here)
  #   if 1 indel in 10bp window: label as forgiven indel
  #   if > 1 indel in 10bp window: label as combination indel
  readwindow = '*' + read[left_idx : right_idx] + '*'
  genomewindow = '*' + genome[left_idx : right_idx] + '*'
  num_dels, num_ins = count_indels(readwindow, genomewindow)
  
  num_indels_in_window = num_dels + num_ins
  if num_indels_in_window == 0:
    return 'combination_indel_notcrispr'
  
  if num_indels_in_window == 1:
    return 'forgiven_indel'
  
  if num_indels_in_window >= 2:
    if num_indels_in_window == total_indels:
      return 'combination_indel'
    else:
      return 'forgiven_combination_indel'
  return 'other'

##
# Categorize
##
def categorize_alignment(read, genome):
  # Alignment endgaps are assumed to be meaningless
  
  # check if homopolymer
  if len(set(read.replace('-', ''))) == 1:
    return 'homopolymer'

  if 'N' in read:
    return 'hasN'

  # Detect PCR recombination within 55bp context (should be rare)
  if detect_pcr_recombination_anywhere(read, genome):
    return 'pcr_recombination'

  # Check if 55bp designed context matches poorly
  if not check_matches_threshold(read, genome):
    return 'poormatches'

  # Categorize wildtype
  if detect_wildtype(read, genome):
    return 'wildtype'

  num_dels, num_ins = count_indels(read, genome)

  # Categorize single deletion
  if num_dels == 1 and num_ins == 0:
    category, score = score_single_deletion(read, genome)
    return category

  # Categorize single insertion
  elif num_dels == 0 and num_ins == 1:
    category, score = score_single_insertion(read, genome)
    return category

  # Categorize combination indel
  elif num_dels + num_ins >= 2:
    return detect_combination_indel(read, genome)
    # can be "forgiven_indel", 
    # "combination_indel",
    # "combination_indel_notcrispr",
    # "other"
  return 'other'

###
# Explore equal-scoring alignments: deletion shifting
###
def shift_single_deletion(read, genome, old_category):
  # Find where cut should be, with possible starting gaps, but exactly one deletion and no insertions in alignment.
  cutsite = get_cutsite_idx(read, genome)
  # seq[:cutsite], seq[cutsite:] gives cut products

  lr = left_or_right_event(read, cutsite)
  if lr == 'left':
    return rightshift_single_deletion(read, genome, old_category, cutsite)
  if lr == 'right':
    return leftshift_single_deletion(read, genome, old_category, cutsite)

def leftshift_single_deletion(read, genome, old_category, cutsite):
  # Deletion is on 3' side of cutsite. Try shifting deletion left.
  # If category improves, accept it
  #   possible progression: del_notcrispr -> del_notbycc -> delX
  del_start = cutsite + read[cutsite:].index('-')
  del_len = trim_start_end_dashes(read).count('-')
  del_end = del_start + del_len

  best_read = read
  best_category = old_category
  best_score = -1
  for jdx in range(del_start - 1, -1, -1):
    shift_len = del_start - jdx
    read_shift = read[jdx:del_start]
    genome_side = genome[del_end - shift_len : del_end] 
    if read_shift == genome_side:
      new_read = read[:jdx] + '-'*del_len + read_shift + read[del_end:]
      new_category, score = score_single_deletion(new_read, genome)
      if score >= best_score:
        best_score = score
        best_read = new_read
        best_category = new_category
    else:
      break
  if del_len in [1, 2] and 'not' in best_category:
    best_category += '_%s' % (del_len)
  return best_read, genome, best_category

def rightshift_single_deletion(read, genome, old_category, cutsite):
  # Deletion is on 5' side of cutsite. Try shifting deletion right
  # If category improves, accept it
  #   possible progression: del_notcrispr -> del_notbycc -> delX
  read_start_idx = read.index(read.replace('-', '')[:1])
  del_start = read_start_idx + read[read_start_idx:].index('-')
  del_len = trim_start_end_dashes(read).count('-')
  del_end = del_start + del_len

  best_read = read
  best_category = old_category
  best_score = -1
  for jdx in range(del_end + 1, len(read)):
    shift_len = jdx - del_end
    read_shift = read[del_end:jdx]
    genome_side = genome[del_start : del_start + shift_len]
    if read_shift == genome_side:
      new_read = read[:del_start] + read_shift + '-'*del_len + read[del_end + shift_len:]
      new_category, score = score_single_deletion(new_read, genome)
      if score >= best_score:
        best_score = score
        best_read = new_read
        best_category = new_category
    else:
      break
  if del_len in [1, 2] and 'not' in best_category:
    best_category += '_%s' % (del_len)
  return best_read, genome, best_category

##
# Explore equal-scoring alignments: insertion shifting
def shift_single_insertion(read, genome, old_category):
  # Find where cut should be, with possible starting gaps, but exactly one deletion and no insertions in alignment.
  cutsite = get_cutsite_idx(read, genome)
  # seq[:cutsite], seq[cutsite:] gives cut products

  lr = left_or_right_event(genome, cutsite)
  if lr == 'left':
    return rightshift_single_insertion(read, genome, old_category, cutsite)
  if lr == 'right':
    return leftshift_single_insertion(read, genome, old_category, cutsite)

def leftshift_single_insertion(read, genome, old_category, cutsite):
  # Insertion is on 3' side of cutsite. Try shifting insertion left
  # If category improves, accept it
  #   possible progression: del_notcrispr -> del_notbycc -> delX
  ins_start = cutsite + genome[cutsite:].index('-')
  ins_len = trim_start_end_dashes(genome).count('-')
  ins_end = ins_start + ins_len

  best_genome = genome
  best_category = old_category
  best_score = -1
  for jdx in range(ins_start - 1, -1, -1):
    shift_len = ins_start - jdx
    genome_shift = genome[jdx:ins_start]
    read_side = read[ins_end - shift_len : ins_end] 
    if genome_shift == read_side:
      new_genome = genome[:ins_start - shift_len] + '-'*ins_len + genome_shift + genome[ins_end:]
      new_category, score = score_single_insertion(read, new_genome)
      if score >= best_score:
        best_genome = new_genome
        best_score = score
        best_category = new_category
    else:
      break
  if ins_len == 1 and 'not' in best_category:
    best_category += '_%s' % (ins_len)
  return read, best_genome, best_category

def rightshift_single_insertion(read, genome, old_category, cutsite):
  # Insertion is on 5' side of cutsite. Try shifting insertion left
  # If category improves, accept it
  #   possible progression: del_notcrispr -> del_notbycc -> delX
  genome_start_idx = genome.index(genome.replace('-', '')[:1])
  ins_start = genome_start_idx + genome[genome_start_idx:].index('-')
  ins_len = trim_start_end_dashes(genome).count('-')
  ins_end = ins_start + ins_len

  best_genome = genome
  best_category = old_category
  best_score = -1
  for jdx in range(ins_end + 1, len(genome)):
    shift_len = jdx - ins_end
    genome_shift = genome[ins_end:jdx]
    read_side = read[ins_start : ins_start + shift_len]
    if genome_shift == read_side:
      new_genome = genome[:ins_start] + genome_shift + '-'*ins_len +  genome[ins_end + shift_len:]
      new_category, score = score_single_insertion(read, new_genome)
      if score >= best_score:
        best_genome = new_genome
        best_score = score
        best_category = new_category
    else:
      break
  if ins_len == 1 and 'not' in best_category:
    best_category += '_%s' % (ins_len)
  return read, best_genome, best_category

###
# Main function
###
def remaster_aligns(inp_fn, data):
  with open(inp_fn) as f:
    for i, line in enumerate(f):
      if i % 4 == 0:
        header = line.strip()
        # read start is 1-based, left-most genomic position of reference in alignment, according to SAM format. 
        read_start = int(header.split('_')[-1])
      if i % 4 == 1:
        read = line.strip()
      if i % 4 == 2:
        genome = line.strip()
        genome = genome.upper()

        # read_start is where reference starts. find expected cutsite by counting along reference until we reach master_expected_cutsite
        global expected_cutsite   # 1-based
        expected_cutsite = master_expected_cutsite - read_start
        if expected_cutsite <= 0 or expected_cutsite >= len(genome.replace('-', '')):
          continue

        # if we reversed this context in our library, reverse the alignment here. adjust expected cutsite accordingly.
        if reverse_flag:
          read = compbio.reverse_complement(read)
          genome = compbio.reverse_complement(genome)
          expected_cutsite = len(genome.replace('-', '')) - expected_cutsite

        # Find where gg should be even if there are insertions in between
        gg_seq = genome[expected_cutsite:].replace('-', '')
        cutsite = get_cutsite_idx(read, genome)
        gg = genome[cutsite:].replace('-', '')[4:6]
        # Assert that cutsite to GG must be in genome
        if len(gg) != 2:
          continue
        assert gg == 'GG', 'No GG!'

        # Ensure that cutsite is determined in a consistent manner between normal and reversed alignments 
        assert genome[cutsite - 1] != '-', 'Inconsistent cutsite!'

        # Main -- Find indel category, assuming end gaps are meaningless
        category = categorize_alignment(read, genome)

        if category in ['del_notatcut', 'del_notcrispr']:
          read, genome, category = shift_single_deletion(read, genome, category)
        if category in ['ins_notatcut', 'ins_notcrispr']:
          read, genome, category = shift_single_insertion(read, genome, category)

        header += '_%s' % (cutsite)
        alignment = [header, read, genome, '']
        data[category] += alignment
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
  for idx in range(3696622, 3702820 + 1, 124):
    start = idx
    end = start + 123
    command = 'python %s.py none %s %s' % (NAME, start, end)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, start)
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

##
# Main
##
@util.time_dec
def main(inp_dir, master_out_dir, srr_id = 'none', start = 'none', end = 'none'):
  # First, ensure output dir is empty
  # Process alignment file, then save.
  print NAME
  util.ensure_dir_exists(master_out_dir)

  # Qsubs
  if srr_id == 'none' and start == 'none' and end == 'none':
    gen_qsubs()
    return

  ## Run single
  elif srr_id != 'none' and start == 'none' and end == 'none':
    print 'Running single...'
    print srr_id
    inp_fn = inp_dir + '%s.txt' % (srr_id)
    out_dir = master_out_dir + srr_id + '/'
    util.ensure_dir_exists(out_dir)

    prepare_align_outdirs(master_out_dir, int(srr_id.replace('SRR', '')), int(srr_id.replace('SRR', '')))

    set_master_expected_cutsite(srr_id)
    data = defaultdict(list)
    remaster_aligns(inp_fn, data)
    save_alignments(data, out_dir)

  elif srr_id == 'none' and start is not 'none' and end is not 'none':
    # Run many
    start, end = int(start), int(end)
    prepare_align_outdirs(master_out_dir, start, end)
    timer = util.Timer(total = end - start + 1)
    for idnum in range(start, end + 1):
      srr_id = 'SRR%s' % (idnum)
      # print srr_id
      inp_fn = inp_dir + '%s.txt' % (srr_id)
      if not os.path.isfile(inp_fn):
        continue
      out_dir = master_out_dir + srr_id + '/'
      util.ensure_dir_exists(out_dir)

      set_master_expected_cutsite(srr_id)
      data = defaultdict(list)
      remaster_aligns(inp_fn, data)
      save_alignments(data, out_dir)
      timer.update()

  return out_dir


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(DEFAULT_INP_DIR, 
         _config.OUT_PLACE + NAME + '/', 
         srr_id = sys.argv[1],
         start = sys.argv[2],
         end = sys.argv[3])
  else:
    main(DEFAULT_INP_DIR, _config.OUT_PLACE + NAME + '/')
