import pandas as pd
from collections import defaultdict
from util import Filenames

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

def train(master_data: dict, filenames: Filenames):
    exps, freqs, dl_freqs = transform_data(master_data)
    ins_stats, bp_stats = get_statistics()

    exps = ['VO-spacers-HEK293-48h-controladj',
            'VO-spacers-K562-48h-controladj',
            'DisLib-mES-controladj',
            'DisLib-U2OS-controladj',
            'Lib1-mES-controladj'
            ] # TODO: REMOVE

    all_rate_stats = pd.DataFrame()
    all_bp_stats = pd.DataFrame()
    for exp in exps:
        # rate_stats = fi2_ins_ratio.load_statistics(exp) TODO
        rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
        # bp_stats = fk_1bpins.load_statistics(exp) TODO

    X, Y, Normalizer = featurize(all_rate_stats, 'Ins1bp/Del Ratio')
    generate_models(X, Y, all_bp_stats, Normalizer)


# =======================================================
# ======= BELOW IS OLD FROM e5_ins_ratebpmodel.py =======
# =======================================================

# from __future__ import division
import pickle
import sys
import numpy as np
from mylib import util
# import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# sys.path.append('/cluster/mshen/')
# # Default params
# DEFAULT_INP_DIR = '/cluster/mshen/prj/mmej_manda2/out/2017-10-27/mb_grab_exons/'
# NAME = util.get_fn(__file__)
# out_dir = "./cluster"


##
# Functions
##
def convert_oh_string_to_nparray(input):
    input = input.replace('[', '').replace(']', '')
    nums = input.split(' ')
    return np.array([int(s) for s in nums])


def featurize(rate_stats, Y_nm):
    fivebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Fivebase_OH']])
    threebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Threebase_OH']])

    ent = np.array(rate_stats['Entropy']).reshape(len(rate_stats['Entropy']), 1)
    del_scores = np.array(rate_stats['Del Score']).reshape(len(rate_stats['Del Score']), 1)
    print(ent.shape, fivebases.shape, del_scores.shape)

    Y = np.array(rate_stats[Y_nm])
    print(Y_nm)

    Normalizer = [(np.mean(fivebases.T[2]),
                   np.std(fivebases.T[2])),
                  (np.mean(fivebases.T[3]),
                   np.std(fivebases.T[3])),
                  (np.mean(threebases.T[0]),
                   np.std(threebases.T[0])),
                  (np.mean(threebases.T[2]),
                   np.std(threebases.T[2])),
                  (np.mean(ent),
                   np.std(ent)),
                  (np.mean(del_scores),
                   np.std(del_scores)),
                  ]

    fiveG = (fivebases.T[2] - np.mean(fivebases.T[2])) / np.std(fivebases.T[2])
    fiveT = (fivebases.T[3] - np.mean(fivebases.T[3])) / np.std(fivebases.T[3])
    threeA = (threebases.T[0] - np.mean(threebases.T[0])) / np.std(threebases.T[0])
    threeG = (threebases.T[2] - np.mean(threebases.T[2])) / np.std(threebases.T[2])
    gtag = np.array([fiveG, fiveT, threeA, threeG]).T

    ent = (ent - np.mean(ent)) / np.std(ent)
    del_scores = (del_scores - np.mean(del_scores)) / np.std(del_scores)

    X = np.concatenate((gtag, ent, del_scores), axis=1)
    # feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
    print('Num. samples: %s, num. features: %s' % X.shape)

    return X, Y, Normalizer


def generate_models(X, Y, bp_stats, Normalizer, filenames: Filenames):
    # Train rate model
    model = KNeighborsRegressor()
    model.fit(X, Y)
    with open(filenames.out_dir + 'rate_model_v2.pkl', 'w') as f:
        pickle.dump(model, f)

    # Obtain bp stats
    bp_model = dict()
    ins_bases = ['A frac', 'C frac', 'G frac', 'T frac']
    t_melt = pd.melt(bp_stats,
                     id_vars=['Base'],
                     value_vars=ins_bases,
                     var_name='Ins Base',
                     value_name='Fraction')
    for base in list('ACGT'):
        bp_model[base] = dict()
        mean_vals = []
        for ins_base in ins_bases:
            crit = (t_melt['Base'] == base) & (t_melt['Ins Base'] == ins_base)
            mean_vals.append(float(np.mean(t_melt[crit])))
        for bp, freq in zip(list('ACGT'), mean_vals):
            bp_model[base][bp] = freq / sum(mean_vals)

    with open(filenames.out_dir + 'bp_model_v2.pkl', 'w') as f:
        pickle.dump(bp_model, f)

    with open(filenames.out_dir + 'Normalizer_v2.pkl', 'w') as f:
        pickle.dump(Normalizer, f)

    return

