from scipy.stats import entropy
import numpy as np
## 
# Shared utility functions useful for general data analysis
##


##
# Subsetting dataframes
##
def notnoise_subset(df):
  notnoise_cats = ['del', 'del_notatcut', 'ins', 'ins_notatcut', 'combination_indel', 'forgiven_indel', 'forgiven_combination_indel', 'wildtype', 'del_notcrispr', 'ins_notcrispr']
  criteria = df['Category'].isin(notnoise_cats)
  return  df[criteria]

def crispr_subset(df):
  crispr_cats = ['del', 'del_notatcut', 'ins', 'ins_notatcut', 'combination_indel']
  criteria = df['Category'].isin(crispr_cats)
  return df[criteria]

def del_subset(df):
  del_cats = ['del', 'del_notatcut', 'del_notcrispr']
  criteria = df['Category'].isin(del_cats)
  return df[criteria]

def ins_subset(df):
  ins_cats = ['ins', 'ins_notatcut', 'ins_notcrispr']
  criteria = df['Category'].isin(ins_cats)
  return df[criteria]

def crispr_del_27bp_subset(df):
  criteria = (df['Category'] == 'del') & (df['Length'] <= 27)
  return df[criteria]

def crispr_del_3bp_27bp_subset(df):
  criteria = (df['Category'] == 'del') & (df['Length'] <= 27) & (df['Length'] >= 3)
  return df[criteria]

def mh_del_subset(df):
  criteria = (df['Category'] == 'del') & (df['Microhomology-Based'] == 'yes')
  return df[criteria]

def indels_without_mismatches_subset(df):
  criteria = (df['Indel with Mismatches'] != 'yes')
  return df[criteria]

def get_sequence_cutsite(df):
  return set(df['_Sequence Context']).pop(), set(df['_Cutsite']).pop()


## 
# Basic Calculations
##
def normalize_frequency(df):
  return df['Count'] / sum(df['Count'])

def normalized_entropy(df):
  if 'Frequency' not in df.columns:
    df['Frequency'] = normalize_frequency[df]
  return entropy(df['Frequency']) / np.log(len(df['Frequency']))

def merge_crispr_events(df1, df2, nm1, nm2):
  join_cols = ['Category', 'Genotype Position', 'Indel with Mismatches', 'Ins Fivehomopolymer', 'Ins Template Length', 'Ins Template Orientation', 'Ins mh2', 'Ins p2', 'Inserted Bases', 'Length', 'Microhomology-Based']
  mdf = df1.merge(df2, how = 'outer', on = join_cols, suffixes = (nm1, nm2))
  mdf['Frequency%s' % (nm1)].fillna(value = 0, inplace = True)
  mdf['Frequency%s' % (nm2)].fillna(value = 0, inplace = True)
  return mdf