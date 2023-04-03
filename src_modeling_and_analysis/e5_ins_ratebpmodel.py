from __future__ import division
import pickle
import sys
import numpy as np
from mylib import util
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import re

sys.path.append('/cluster/mshen/')
# Default params
DEFAULT_INP_DIR = '/cluster/mshen/prj/mmej_manda2/out/2017-10-27/mb_grab_exons/'
NAME = util.get_fn(__file__)
out_dir = "/cluster/mshen/prj/mmej_figures/out/e5_ins_ratebpmodel/"


##
# Functions
##
def convert_oh_string_to_nparray(input):
    input = str(input).replace('[', '').replace(']', '')
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


def generate_models(X, Y, bp_stats, Normalizer):
    # Train rate model
    model = KNeighborsRegressor()
    model.fit(X, Y)
    with open(out_dir + 'rate_model_v2.pkl', 'wb') as f:
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

    with open(out_dir + 'bp_model_v2.pkl', 'wb') as f:
        print(f"saved bp model as {out_dir}bp_model_v2.pkl")
        pickle.dump(bp_model, f)

    with open(out_dir + 'Normalizer_v2.pkl', 'wb') as f:
        print(f"saved normalizer as {out_dir}Normalizer_v2.pkl")
        pickle.dump(Normalizer, f)

    return


##
# Main
##
@util.time_dec
def main(data_nm=''):
    # print(NAME)
    # out_place = './cluster/mshen/prj/mmej_figures/out/d2_model/'
    import fi2_ins_ratio
    import fk_1bpins
    global out_dir
    out_dir = "." + out_dir
    util.ensure_dir_exists(out_dir)

    # ========
    master_data = pickle.load(open("../pickle_data/inDelphi_counts_and_deletion_features.pkl", "rb"))

    res = master_data['counts']
    res['key_0'] = res.index
    # del_features = master_data['del_features']
    # res = pd.merge(master_data['counts'], master_data['del_features'],
    #                left_on=master_data['counts'].index, right_on=master_data['del_features'].index)
    res[['sample', 'offset']] = pd.DataFrame(res['key_0'].tolist(), index=res.index)
    res = res[res['Type'] == 'INSERTION']

    # mh_lens = []
    # gc_fracs = []
    # del_lens = []
    exps = []
    freqs = []
    dl_freqs = []
    for group in res.groupby("sample"):
        # mh_lens.append(group[1]['homologyLength'].values)
        # gc_fracs.append(group[1]['homologyGCContent'].values)
        # del_lens.append(group[1]['Size'].values)
        exps.append(group[1]['key_0'].values[0][0])
        freqs.append(group[1]['countEvents'].values)
        dl_freqs.append(group[1]['fraction'].values)

    # TODO: I think the author hardcoded this and we don't want to do that?

    # ========
    # exps = ['VO-spacers-HEK293-48h-controladj',
    #         'VO-spacers-K562-48h-controladj',
    #         'DisLib-mES-controladj',
    #         'DisLib-U2OS-controladj',
    #         'Lib1-mES-controladj'
    #         ]

    all_rate_stats = pd.DataFrame()
    all_bp_stats = pd.DataFrame()
    # filter for testing purposes, use exps if you want the whole dataset
    # (which is enormous and takes a long while to train)
    # filtered_exps = list(filter(lambda x: re.search("[a-z]", x), exps))
    for exp in exps:
        rate_stats = fi2_ins_ratio.load_statistics(exp)
        rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
        bp_stats = fk_1bpins.load_statistics(exp)
        # exps = rate_stats['_Experiment']
        #
        #
        if 'DisLib' in exp:
            crit = (rate_stats['_Experiment'] >= 73) & (rate_stats['_Experiment'] <= 300)
            rs = rate_stats[crit]
            all_rate_stats.append(rs, ignore_index=True)

            crit = (rate_stats['_Experiment'] >= 16) & (rate_stats['_Experiment'] <= 72)
            rs = rate_stats[crit]
            rs = rs[rs['Ins1bp Ratio'] < 0.3]  # remove outliers
            all_rate_stats.append(rs, ignore_index=True)

            crit = (bp_stats['_Experiment'] >= 73) & (bp_stats['_Experiment'] <= 300)
            rs = bp_stats[crit]
            all_bp_stats.append(rs, ignore_index=True)

            crit = (bp_stats['_Experiment'] >= 16) & (bp_stats['_Experiment'] <= 72)
            rs = bp_stats[crit]
            all_bp_stats.append(rs, ignore_index=True)

        all_rate_stats = pd.concat([all_rate_stats, rate_stats], ignore_index=True)
        all_bp_stats = pd.concat([all_bp_stats, bp_stats], ignore_index=True)

        # print(exp, len(all_rate_stats))

    # TODO: check if this makes sense
    # all_rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
    X, Y, Normalizer = featurize(all_rate_stats, 'Ins1bp/Del Ratio')
    generate_models(X, Y, all_bp_stats, Normalizer)

    return


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(data_nm=sys.argv[1])
    else:
        main()
