from __future__ import division

import copy
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import entropy

import d2_model as model

from inDelphi.util import split_data_set


# global vars
nn_params = None
nn2_params = None
test_exps = None


##
# Sequence featurization
##
def get_gc_frac(seq):
    return (seq.count('C') + seq.count('G')) / len(seq)


def find_microhomologies(left, right):
    start_idx = max(len(right) - len(left), 0)
    mhs = []
    mh = [start_idx]
    for idx in range(min(len(right), len(left))):
        if left[idx] == right[start_idx + idx]:
            mh.append(start_idx + idx + 1)
        else:
            mhs.append(mh)
            mh = [start_idx + idx + 1]
    mhs.append(mh)
    return mhs


def featurize(seq, cutsite, DELLEN_LIMIT=60):
    # print 'Using DELLEN_LIMIT = %s' % (DELLEN_LIMIT)
    mh_lens, gc_fracs, gt_poss, del_lens = [], [], [], []
    for del_len in range(1, DELLEN_LIMIT):
        left = seq[cutsite - del_len: cutsite]
        right = seq[cutsite: cutsite + del_len]

        mhs = find_microhomologies(left, right)
        for mh in mhs:
            mh_len = len(mh) - 1
            if mh_len > 0:
                gtpos = max(mh)
                gt_poss.append(gtpos)

                s = cutsite - del_len + gtpos - mh_len
                e = s + mh_len
                mh_seq = seq[s: e]
                gc_frac = get_gc_frac(mh_seq)

                mh_lens.append(mh_len)
                gc_fracs.append(gc_frac)
                del_lens.append(del_len)

    return mh_lens, gc_fracs, gt_poss, del_lens


##
# Prediction
##
def predict_all(seq, cutsite, rate_model, bp_model, normalizer):
    # Predict 1 bp insertions and all deletions (MH and MH-less)
    # Most complete "version" of inDelphi
    # Requires rate_model (k-NN) to predict 1 bp insertion rate compared to deletion rate
    # Also requires bp_model to predict 1 bp insertion genotype given -4 nucleotide

    ################################################################
    #####
    ##### Predict MH and MH-less deletions
    #####
    # Predict MH deletions

    mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)

    # Form inputs
    pred_input = np.array([mh_len, gc_frac]).T
    del_lens = np.array(del_len).T

    # Predict
    mh_scores = model.nn_match_score_function(nn_params, pred_input)
    mh_scores = mh_scores.reshape(mh_scores.shape[0], 1)
    Js = del_lens.reshape(del_lens.shape[0], 1)
    unfq = np.exp(mh_scores - 0.25 * Js)

    # Add MH-less contribution at full MH deletion lengths
    mh_vector = np.array(mh_len)
    mhfull_contribution = np.zeros(mh_vector.shape)
    for jdx in range(len(mh_vector)):
        if del_lens[jdx] == mh_vector[jdx]:
            dl = del_lens[jdx]
            mhless_score = model.nn_match_score_function(nn2_params, np.array(dl))
            mhless_score = np.exp(mhless_score - 0.25 * dl)
            mask = np.concatenate([np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
            mhfull_contribution = mhfull_contribution + mask
    mhfull_contribution = mhfull_contribution.reshape(-1, 1)
    unfq = unfq + mhfull_contribution

    # Store predictions to combine with mh-less deletion preds
    pred_del_len = copy.copy(del_len)
    pred_gt_pos = copy.copy(gt_pos)

    ################################################################
    #####
    ##### Predict MH and MH-less deletions
    #####
    # Predict MH-less deletions
    mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)

    unfq = list(unfq)

    pred_mhless_d = defaultdict(list)
    # Include MH-less contributions at non-full MH deletion lengths
    nonfull_dls = []
    for dl in range(1, 60):
        if dl not in del_len:
            nonfull_dls.append(dl)
        elif del_len.count(dl) == 1:
            idx = del_len.index(dl)
            if mh_len[idx] != dl:
                nonfull_dls.append(dl)
        else:
            nonfull_dls.append(dl)

    mh_vector = np.array(mh_len)
    for dl in nonfull_dls:
        mhless_score = model.nn_match_score_function(nn2_params, np.array(dl))
        mhless_score = np.exp(mhless_score - 0.25 * dl)

        unfq.append(mhless_score)
        pred_gt_pos.append('e')
        pred_del_len.append(dl)

    unfq = np.array(unfq)
    total_phi_score = float(sum(unfq))

    nfq = np.divide(unfq, np.sum(unfq))
    pred_freq = list(nfq.flatten())

    d = {'Length': pred_del_len, 'Genotype Position': pred_gt_pos, 'Predicted_Frequency': pred_freq}
    pred_del_df = pd.DataFrame(d)
    pred_del_df['Category'] = 'del'

    ################################################################
    #####
    ##### Predict Insertions
    #####
    # Predict 1 bp insertions
    del_score = total_phi_score
    dlpred = []
    for dl in range(1, 28 + 1):
        crit = (pred_del_df['Length'] == dl)
        dlpred.append(sum(pred_del_df[crit]['Predicted_Frequency']))
    dlpred = np.array(dlpred) / sum(dlpred)
    norm_entropy = entropy(dlpred) / np.log(len(dlpred))

    # feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
    fiveohmapper = {'A': [0, 0],
                    'C': [0, 0],
                    'G': [1, 0],
                    'T': [0, 1]}
    threeohmapper = {'A': [1, 0],
                     'C': [0, 0],
                     'G': [0, 1],
                     'T': [0, 0]}
    fivebase = seq[cutsite - 1]
    threebase = seq[cutsite]
    onebp_features = fiveohmapper[fivebase] + threeohmapper[threebase] + [norm_entropy] + [del_score]
    for idx in range(len(onebp_features)):
        val = onebp_features[idx]
        onebp_features[idx] = (val - normalizer[idx][0]) / normalizer[idx][1]
    onebp_features = np.array(onebp_features).reshape(1, -1)
    rate_1bpins = float(rate_model.predict(onebp_features))

    # Predict 1 bp genotype frequencies
    pred_1bpins_d = defaultdict(list)
    for ins_base in bp_model[fivebase]:
        freq = bp_model[fivebase][ins_base]
        freq *= rate_1bpins / (1 - rate_1bpins)

        pred_1bpins_d['Category'].append('ins')
        pred_1bpins_d['Length'].append(1)
        pred_1bpins_d['Inserted Bases'].append(ins_base)
        pred_1bpins_d['Predicted_Frequency'].append(freq)

    pred_1bpins_df = pd.DataFrame(pred_1bpins_d)
    pred_all_df = pred_del_df.append(pred_1bpins_df, ignore_index=True)
    pred_all_df['Predicted_Frequency'] /= sum(pred_all_df['Predicted_Frequency'])

    return pred_del_df, pred_all_df, total_phi_score, rate_1bpins


##
# Init
##
def init_model(run_iter='aey', param_iter='abo'):
    # run_iter = 'aav'
    # param_iter = 'aag'
    # run_iter = 'aaw'
    # param_iter = 'aae'
    # run_iter = 'aax'
    # param_iter = 'aag'
    # run_iter = 'aay'
    # param_iter = 'aae'

    print('Initializing model %s/%s...' % (run_iter, param_iter))

    model_out_dir = '/cluster/mshen/prj/mmej_figures/out/d2_model/'

    param_fold = model_out_dir + '%s/parameters/' % run_iter
    global nn_params
    global nn2_params
    nn_params = pickle.load(open("./" + param_fold + '%s_nn.pkl' % param_iter, "rb"))
    nn2_params = pickle.load(open("./" + param_fold + '%s_nn2.pkl' % param_iter, "rb"))

    print('Done')
    return


def init_rate_bp_models():
    model_dir = '/cluster/mshen/prj/mmej_figures/out/e5_ins_ratebpmodel/'

    rate_model_nm = 'rate_model_v2'
    bp_model_nm = 'bp_model_v2'
    normalizer_nm = 'Normalizer_v2'

    print('Loading %s...\nLoading %s...' % (rate_model_nm, bp_model_nm))
    with open("." + model_dir + '%s.pkl' % rate_model_nm, "rb") as f:
        rate_model = pickle.load(f)
    with open("." + model_dir + '%s.pkl' % bp_model_nm, "rb") as f:
        bp_model = pickle.load(f)
    with open("." + model_dir + '%s.pkl' % normalizer_nm, "rb") as f:
        normalizer = pickle.load(f)
    return rate_model, bp_model, normalizer


if __name__ == "__main__":
    init_model()
    rate_model, bp_model, normalizer = init_rate_bp_models()
    dataset = pickle.load(open("../pickle_data/inDelphi_counts_and_deletion_features.pkl", "rb"))
    training_data, test_data = split_data_set(dataset)
    dataset = pd.merge(training_data['counts'], training_data['del_features'],
                       left_on=training_data['counts'].index, right_on=training_data['del_features'].index,
                       how="left")
    dataset['Length'] = [int(x[1].split("+")[1]) if x[1].split("+")[1].isdigit() else len(x[1].split("+")[1]) for x in
                         dataset["key_0"]]
    dataset['cutSite'] = [int(x[1].split("+")[0]) + 29 for x in dataset["key_0"]]
    dataset['exp'] = [x[0] for x in dataset["key_0"]]

    # TODO ignores cutsites not compatible with liba, is this correct?
    dataset = dataset[dataset['cutSite'] > 4]
    with open("../data_libprocessing/targets-libA.txt") as f:
        full_dna_exps = []
        for line in f:
            full_dna_exps.append(line.strip("\n"))

    exps = list(set(dataset['exp']))

    exps = exps[0:20] #select fewer exps for testing purposes
    for i, exp in enumerate(exps):
        print(i)
        header_data = list(dataset[dataset["exp"] == exp]["exp"])[0].split("_")[:-1]
        header = ""
        for h in header_data:
            header = header + h + "_"

        header.removesuffix("_")

        exp_substring_dna = exp[-20:]
        matched_dna = list(filter(lambda x: exp_substring_dna in x, full_dna_exps))
        if matched_dna:
            sequence = matched_dna[0]
        else:
            print(f"Experiment {exp} not in libA!")
            continue
        cutsites = dataset[dataset["exp"] == exp]["cutSite"]
        for cutsite in cutsites:
            dataframes = predict_all(sequence, cutsite, rate_model, bp_model, normalizer)
