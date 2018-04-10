# Config parameters
# imported by src/_config

NAMES = ['GH', 'IJ']

data_dir = '/cluster/mshen/prj/mmej_manda/data/2017-08-23/'

GRNAS = open(data_dir + 'grna.txt').read().split()
TARGETS = open(data_dir + 'targets.txt').read().split()
# TARGETS_EXPWT = open(data_dir + 'targets_expwt.txt').read().split()
OLIGO_NAMES = open(data_dir + 'names.txt').read().split()

def add_mer(inp):
  new = []
  for mer in inp:
    for nt in ['A', 'C', 'G', 'T']:
      new.append(mer + nt)
  return new

threemers = add_mer(add_mer(['A', 'C', 'G', 'T']))