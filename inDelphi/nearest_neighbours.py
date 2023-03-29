import pandas as pd
from collections import defaultdict


def prepare_statistics():
    bp_ins = defaultdict(list)
def get_statistics():
    ins_stats, bp_stats = prepare_statistics()


def transform_data(master_data):
    res = master_data['counts']
    res['key_0'] = res.index
    res[['sample', 'offset']] = pd.DataFrame(res['key_0'].tolist(), index=res.index)
    res = res[res['Type'] == 'INSERTION']

    expectations = []
    freqs = []
    dl_freqs = []
    for group in res.groupby("sample"):
        expectations.append(group[1]['key_0'].values)
        freqs.append(group[1]['countEvents'].values)
        dl_freqs.append(group[1]['fraction'].values)

    return expectations, freqs, dl_freqs

def train(master_data: dict):
    exps, freqs, dl_freqs = transform_data(master_data)
    # ins_stats, bp_stats = get_statistics()