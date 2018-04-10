import pandas as pd
curr_dir = '/cluster/mshen/prj/vanoverbeek/data/'

TABLE = pd.read_csv(curr_dir + 'overbeek_data.csv')
RUNS_SET = set(TABLE['Run'])

def sam_fn(srr_id):
  # ex: /cluster/mshen/data/SRP076796/SRR3696/SRR3696622.sam
  prefix = srr_id[:7]
  return '/cluster/mshen/data/SRP076796/%s/%s.sam' % (prefix, srr_id)