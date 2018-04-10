import pandas as pd
from collections import defaultdict
from mylib import util
from mylib import compbio


data_dir = '/cluster/mshen/prj/mmej_figures/data/'

targets = open(data_dir + 'targets_dislib.txt').readlines() 
names = open(data_dir + 'names_dislib.txt').readlines()

def convert_lib_nm(nm):
  nm = nm.replace('clinvar_10_rest_', '')
  nm = nm.replace('clinvar_10_best_', '')
  filt_num = False
  if 'hgmd' in nm:
    filt_num = True
  else:
    if nm[-2] == '_':
      nm = nm[:-2]
  nm = nm.replace('hgmd_rest_', '')
  nm = nm.replace('hgmd_best_', '')
  if filt_num:
    nm = nm.split('_')[0]
  nm = nm.replace('^', ' ')
  return nm

def combine_clinvars_10_cc():
  data10 = pd.read_csv(data_dir + 'clinvars_mmej_10.csv')
  datacc = pd.read_csv(data_dir + 'clinvars_mmej_cc.csv')
  data = defaultdict(list)
  for idx, row in data10.iterrows():
    nm = row['Name']
    cs = row['Cutsite']
    data[nm].append( cs )
  for idx, row in datacc.iterrows():
    nm = row['Name']
    cs = row['Cutsite']
    if cs not in data[nm]:
      data10 = data10.append( row , ignore_index = True )
  return data10

def classify_exp(exp):
  if exp < 8:
    return 'str_iter'
  if exp < 16:
    return 'str_noiter'
  if exp < 73:
    return 'longdup'
  if exp < 301:
    return 'fourbp'
  if exp < 391:
    return 'overbeek'
  if exp < 408:
    return 'manda'
  if exp < 650:
    return 'clinvar-best'
  if exp < 1735:
    return 'hgmd-best'
  if exp < 1780:
    return 'clinvar-rest'
  return 'hgmd-rest'

def find_matching_sequence(target, rows):
  for idx, row in rows.iterrows():
    orient = row['gRNA Orientation']
    seq = row['Alternative Sequence']
    cutsite = row['Cutsite']
    if orient == '-':
      seq = compbio.reverse_complement(seq)
      cutsite = len(seq) - cutsite
    cons_target = seq[cutsite - 27 : cutsite + 28]
    if target == cons_target:
      return row
  assert False, 'Not found'
  return

def find_microhomologies(left, right):
  start_idx = max(len(right) - len(left), 0)
  mhs = []
  mh = [start_idx]
  for idx in range(min(len(right), len(left))):
    if left[idx] == right[start_idx + idx]:
      mh.append(start_idx + idx + 1)
    else:
      mhs.append(mh)
      mh = [start_idx + idx +1]
  mhs.append(mh)
  return mhs

def wildtype_repairs(row):
  orient = row['gRNA Orientation']
  cutsite = row['Cutsite']
  seq = row['Alternative Sequence']
  wt_seq = row['Reference Sequence']
  if orient == '-':
    seq = compbio.reverse_complement(seq)
    wt_seq = compbio.reverse_complement(wt_seq)
    cutsite = len(seq) - cutsite

  # Detect wildtypes with iterative cutting - expect 0 at these
  wt_repairable_flag = 'yes'
  fs_repairable_flag = 'yes'
  grna = seq[cutsite-10:cutsite+3]
  for wt_seq_s in [wt_seq, compbio.reverse_complement(wt_seq)]:
    if grna in wt_seq_s:
      try:
        pam = wt_seq[wt_seq.index(grna)+14 : wt_seq.index(grna)+16]
      except:
        wt_repairable_flag = 'iterwt'
        fs_repairable_flag = 'iterwt'
        continue
      if pam in ['GG', 'AG', 'GA']:
        wt_repairable_flag = 'iterwt'
        fs_repairable_flag = 'iterwt'

  repair_gts = []
  repair_dls = []
  longest_wt_mh = -1
  longest_nonwt_mh = -1
  for del_len in range(1, 27+1):
    for start_pos in range(0, del_len + 1):
      repair_gt = seq[:cutsite - del_len + start_pos] + seq[cutsite + start_pos:]
      l = seq[cutsite - del_len : cutsite]
      r = seq[cutsite : cutsite + del_len]
      mhs = find_microhomologies(l, r)
      if repair_gt == wt_seq:
        repair_gts.append(start_pos)
        repair_dls.append(del_len)
        for mh in mhs:
          if start_pos in mh:
            mh_len = len(mh) - 1
            if mh_len > longest_wt_mh:
              longest_wt_mh = mh_len
      else:
        for mh in mhs:
          if start_pos in mh:
            mh_len = len(mh) - 1
            if mh_len > longest_nonwt_mh:
              longest_nonwt_mh = mh_len
  if len(repair_gts) == 0:
    wt_repairable_flag = 'no'

  if longest_wt_mh > longest_nonwt_mh:
    longest_mh_wt = 'yes'
  else:
    longest_mh_wt = 'no'    

  fs = row['Needed Frameshift']
  if fs == 0:
    fs_repairable_flag = 'no'
  return repair_gts, repair_dls, wt_repairable_flag, fs, fs_repairable_flag, longest_mh_wt



##
# Main
##
def main():
  clinvar_data = combine_clinvars_10_cc()
  hgmd_data = pd.read_csv(data_dir + 'hgmd_extract_analysis.csv')

  dd = defaultdict(list)


  timer = util.Timer(total = len(names) - 408 + 1)
  for idx in range(408, len(names)):
    nm = names[idx].strip()
    context = targets[idx].strip()
    grna = context[10:30]

    exp_type = classify_exp(idx)
    if 'clinvar' in exp_type:
      data = clinvar_data
    elif 'hgmd' in exp_type:
      data = hgmd_data
    
    clinvar_cols = ['Indel Type',
              'Reference Sequence',
              'Alternative Sequence',
              '#AlleleID',
              'Assembly',
              'Chromosome',
              'ChromosomeAccession',
              'ClinicalSignificance',
              'Cytogenetic',
              'GeneID',
              'GeneSymbol',
              'HGNC_ID',
              'LastEvaluated',
              'NumberSubmitters',
              'Origin',
              'OtherIDs',
              'PhenotypeIDS',
              'PhenotypeList',
              'RCVaccession',
              'RS_num_dbSNP',
              'ReviewStatus',
              'Start',
              'Stop',
              'gRNA',
              'gRNA Orientation',
              'gRNA Start (Alt Sequence)',
              'gRNA End (Alt Sequence)',
              'gRNA Cas9 Type',
              'heterozygous selection estimate (higher is worse)', 
              'selection decile (higher is worse)', 
              'OMIM ID If Easy to extract',
              'OMIM Phenotype',
              ]
    hgmd_cols = ['omimid',
              'gdbid',
              'genename',
              'chrom',
              'gene',
              'disease',
              'gRNA Cas9 Type',
              'gRNA Orientation',
              'Reference Sequence',
              'Alternative Sequence',
              'new_date',
              'acc_num',
              'comments',
              'reftag',
              'pmid',
              'year',
              'page',
              'vol',
              'allname',
              'fullname',
              'author',
              'tag',
              'Stop',
              'Start',
              'dbsnp',
              'hgvsAll',
              'hgvs',
              'descr',
              'codonAff',
              'codon',
              'insertion',
              'deletion',
              ]

    # Find proper row
    conv_nm = convert_lib_nm(nm)
    rows = data[data['Name'] == conv_nm]
    row = find_matching_sequence(context, rows)

    # Add metadata
    if 'hgmd' in exp_type:
      for col in hgmd_cols:
        dd[col].append(row[col])
      intersection_cols = [s for s in clinvar_cols if s not in hgmd_cols]
      for ic in intersection_cols:
        dd[ic].append('')

    elif 'clinvar' in exp_type:
      for col in clinvar_cols:
        dd[col].append(row[col])
      intersection_cols = [s for s in hgmd_cols if s not in clinvar_cols]
      for ic in intersection_cols:
        dd[ic].append('')


    repair_gts, repair_dls, wt_repairable, fs, fs_repairable, longest_mh_wt = wildtype_repairs(row)

    dd['longestmh_is_wt'].append(longest_mh_wt)
    dd['wt_repairable'].append(wt_repairable)
    dd['fs_repairable'].append(fs_repairable)
    dd['index'].append(idx)
    dd['name'].append(nm)
    dd['dls'].append(';'.join([str(s) for s in repair_dls]))
    dd['gts'].append(';'.join([str(s) for s in repair_gts]))
    dd['fs'].append(fs)

    timer.update()

  import code; code.interact(local=dict(globals(), **locals()))
  df = pd.DataFrame(dd)
  df.to_csv(data_dir + 'dislib_wt.csv')
  return


if __name__ == '__main__':
  main()