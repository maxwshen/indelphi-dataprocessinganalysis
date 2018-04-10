import sys

PRJ_DIR = '/cluster/mshen/prj/vanoverbeek/'  
SRC_DIR = PRJ_DIR + 'src/'

# toy = True
toy = False
if toy:
  PRJ_DIR += 'toy/'
#######################################################
# Note: Directories should end in / always
#######################################################
DATA_DIR = PRJ_DIR + 'data/'
OUT_PLACE = PRJ_DIR + 'out/'
RESULTS_PLACE = PRJ_DIR + 'results/'
QSUBS_DIR = PRJ_DIR + 'qsubs/'
#######################################################
#######################################################

CLEAN = False       # Values = 'ask', True, False

# which data are we using? import that data's parameters
# DATA_FOLD = 'rename_me/'
DATA_FOLD = ''

sys.path.insert(0, DATA_DIR + DATA_FOLD)
import _dataconfig as d
print 'Using data folder:\n', DATA_DIR + DATA_FOLD
DATA_DIR += DATA_FOLD
OUT_PLACE += DATA_FOLD
RESULTS_PLACE += DATA_FOLD
QSUBS_DIR += DATA_FOLD

#######################################################
# Project-specific parameters
#######################################################

