from __future__ import division
import pickle, imp
import copy
import numpy as np
from collections import defaultdict
import pandas as pd
from scipy.stats import entropy


# global vars
model = None
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
      mh = [start_idx + idx +1]
  mhs.append(mh)
  return mhs

def featurize(seq, cutsite, DELLEN_LIMIT = 60):
  # print 'Using DELLEN_LIMIT = %s' % (DELLEN_LIMIT)
  mh_lens, gc_fracs, gt_poss, del_lens = [], [], [], []
  for del_len in range(1, DELLEN_LIMIT):
    left = seq[cutsite - del_len : cutsite]
    right = seq[cutsite : cutsite + del_len]

    mhs = find_microhomologies(left, right)
    for mh in mhs:
      mh_len = len(mh) - 1
      if mh_len > 0:
        gtpos = max(mh)
        gt_poss.append(gtpos)

        s = cutsite - del_len + gtpos - mh_len
        e = s + mh_len
        mh_seq = seq[s : e]
        gc_frac = get_gc_frac(mh_seq)

        mh_lens.append(mh_len)
        gc_fracs.append(gc_frac)
        del_lens.append(del_len)

  return mh_lens, gc_fracs, gt_poss, del_lens

def featurize_cpf1(seq, cutsite, DELLEN_LIMIT = 60):
  print 'Using DELLEN_LIMIT = %s' % (DELLEN_LIMIT)
  mh_lens, gc_fracs, gt_poss, del_lens = [], [], [], []
  for del_len in range(5, DELLEN_LIMIT):
    left = seq[cutsite - del_len + 4 : cutsite]
    right = seq[cutsite + 4 : cutsite + del_len]

    mhs = find_microhomologies(left, right)
    for mh in mhs:
      mh_len = len(mh) - 1
      if mh_len > 0:
        gtpos = max(mh) + 4
        gt_poss.append(gtpos)

        s = cutsite - del_len + gtpos - mh_len
        e = s + mh_len
        mh_seq = seq[s : e]
        assert mh_seq == seq[s + del_len : e + del_len]
        gc_frac = get_gc_frac(mh_seq)

        mh_lens.append(mh_len)
        gc_fracs.append(gc_frac)
        del_lens.append(del_len)

  return mh_lens, gc_fracs, gt_poss, del_lens

##
# Prediction
##
def predict_mhdel(seq, cutsite):
  # Simulates all microhomology-based deletions from sequence and cutsite.
  # Returns a dataframe of the predicted frequencies of all possible mh-based deletions. 
  if model is None:
    print 'ERROR: Model not initialized'
    return None

  # Extract features from sequence, cutsite
  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite)

  # Form inputs
  pred_input = np.array([mh_len, gc_frac]).T
  del_lens = np.array(del_len).T
  
  ##
  # MH-based deletion frequencies
  ##
  mh_scores = model.nn_match_score_function(nn_params, pred_input)
  mh_scores = mh_scores.reshape(mh_scores.shape[0], 1)
  Js = del_lens.reshape(del_lens.shape[0], 1)
  unnormalized_fq = np.exp(mh_scores - 0.25*Js)
  unnormalized_fq = unnormalized_fq.flatten()

  # Add MH-less contribution at full MH deletion lengths
  mh_vector = np.array(mh_len)
  mhfull_contribution = np.zeros(mh_vector.shape)
  for jdx in range(len(mh_vector)):
    if del_lens[jdx] == mh_vector[jdx]:
      dl = del_lens[jdx]
      mhless_score = model.nn_match_score_function(nn2_params, np.array(dl))
      mhless_score = np.exp(mhless_score - 0.25*dl)
      mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
      mhfull_contribution = mhfull_contribution + mask
  unnormalized_fq = unnormalized_fq + mhfull_contribution
  normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

  pred_freq = list(normalized_fq.flatten())

  # Form dataframe
  d = {'Length': del_len, 'Genotype Position': gt_pos, 'Predicted_Frequency': pred_freq}
  pred_df = pd.DataFrame(d)
  pred_df['Category'] = 'del'

  return pred_df

def predict_indels(seq, cutsite, rate_model, bp_model):
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
  unfq = np.exp(mh_scores - 0.25*Js)

  # Add MH-less contribution at full MH deletion lengths
  mh_vector = np.array(mh_len)
  mhfull_contribution = np.zeros(mh_vector.shape)
  for jdx in range(len(mh_vector)):
    if del_lens[jdx] == mh_vector[jdx]:
      dl = del_lens[jdx]
      mhless_score = model.nn_match_score_function(nn2_params, np.array(dl))
      mhless_score = np.exp(mhless_score - 0.25*dl)
      mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
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
    mhless_score = np.exp(mhless_score - 0.25*dl)

    unfq.append(mhless_score)
    pred_gt_pos.append('e')
    pred_del_len.append(dl)

  unfq = np.array(unfq)
  nfq = np.divide(unfq, np.sum(unfq))  
  pred_freq = list(nfq.flatten())

  d = {'Length': pred_del_len, 'Genotype Position': pred_gt_pos, 'Predicted_Frequency': pred_freq}
  pred_df = pd.DataFrame(d)
  pred_df['Category'] = 'del'

  ################################################################
  #####
  ##### Predict Insertions
  #####
  # Predict 1 bp insertions
  del_score = total_deletion_score(seq, cutsite)
  dlpred = deletion_length_distribution(seq, cutsite)
  norm_entropy = entropy(dlpred) / np.log(len(dlpred))
  ohmapper = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
  fivebase = seq[cutsite - 1]
  onebp_features = ohmapper[fivebase] + [norm_entropy] + [del_score]
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
  pred_df = pred_df.append(pred_1bpins_df, ignore_index = True)
  pred_df['Predicted_Frequency'] /= sum(pred_df['Predicted_Frequency'])

  return pred_df

def predict_mhdel_cpf1(seq, cutsite):
  # Simulates all microhomology-based deletions from sequence and cutsite.
  # Returns a dataframe of the predicted frequencies of all possible mh-based deletions. 
  if model is None:
    print 'ERROR: Model not initialized'
    return None

  # Extract features from sequence, cutsite
  mh_len, gc_frac, gt_pos, del_len = featurize_cpf1(seq, cutsite, DELLEN_LIMIT = 60)

  # Form inputs
  pred_input = np.array([mh_len, gc_frac]).T
  del_lens = np.array(del_len).T
  
  ##
  # MH-based deletion frequencies
  ##
  mh_scores = model.nn_match_score_function(nn_params, pred_input)
  mh_scores = mh_scores.reshape(mh_scores.shape[0], 1)
  Js = del_lens.reshape(del_lens.shape[0], 1)
  unnormalized_fq = np.exp(mh_scores - 0.15*(Js - 4))
  unnormalized_fq = unnormalized_fq.flatten()
  normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

  pred_freq = list(normalized_fq.flatten())

  # Form dataframe
  d = {'Length': del_len, 'Genotype Position': gt_pos, 'Predicted_Frequency': pred_freq}
  pred_df = pd.DataFrame(d)
  pred_df['Category'] = 'del'

  return pred_df

def deletion_length_distribution(seq, cutsite):
  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite, DELLEN_LIMIT = 28)

  # Form inputs
  pred_input = np.array([mh_len, gc_frac]).T
  del_lens = np.array(del_len).T
  
  # Predict
  mh_scores = model.nn_match_score_function(nn_params, pred_input)
  mh_scores = mh_scores.reshape(mh_scores.shape[0], 1)
  Js = del_lens.reshape(del_lens.shape[0], 1)
  unfq = np.exp(mh_scores - 0.25*Js)

  dls = np.arange(1, 28+1)
  dls = dls.reshape(28, 1)
  nn2_scores = model.nn_match_score_function(nn2_params, dls)
  unfq2 = np.exp(nn2_scores - 0.25*np.arange(1, 28+1))

  for ufq, dl in zip(unfq, Js):
    unfq2[int(dl)-1] += float(ufq)

  nfq = unfq2 / sum(unfq2)
  return nfq

def total_deletion_score(seq, cutsite):
  # Get the total unnormalized score.
  # Could be useful for predicting the frequency of repair classes
  mh_len, gc_frac, gt_pos, del_len = featurize(seq, cutsite, DELLEN_LIMIT = 28)

  # Form inputs
  pred_input = np.array([mh_len, gc_frac]).T
  del_lens = np.array(del_len).T
  
  # Predict
  mh_scores = model.nn_match_score_function(nn_params, pred_input)
  mh_scores = mh_scores.reshape(mh_scores.shape[0], 1)
  Js = del_lens.reshape(del_lens.shape[0], 1)
  unfq = np.exp(mh_scores - 0.25*Js)

  dls = np.arange(1, 28+1)
  dls = dls.reshape(28, 1)
  nn2_scores = model.nn_match_score_function(nn2_params, dls)
  unfq2 = np.exp(nn2_scores - 0.25*np.arange(1, 28+1))

  return float(sum(unfq) + sum(unfq2))

##
# Init
##
def init_model(run_iter = 'aax', param_iter = 'aag'):
  # run_iter = 'aav'
  # param_iter = 'aag'
  # run_iter = 'aaw'
  # param_iter = 'aae'
  # run_iter = 'aax'
  # param_iter = 'aag'
  # run_iter = 'aay'
  # param_iter = 'aae'
  global model
  if model != None:
    return

  print 'Initializing model %s/%s...' % (run_iter, param_iter)
  
  model_out_dir = '/cluster/mshen/prj/mmej_figures/out/d2_model/'

  param_fold = model_out_dir + '%s/parameters/' % (run_iter)
  global nn_params
  global nn2_params
  nn_params = pickle.load(open(param_fold + '%s_nn.pkl' % (param_iter)))
  nn2_params = pickle.load(open(param_fold + '%s_nn2.pkl' % (param_iter)))

  model = imp.load_source('model', model_out_dir + '%s/d2_model.py' % (run_iter))
  
  # test_df = pd.read_csv(model_out_dir + '%s/%s_test_rsqs_params.csv' % (run_iter, param_iter))
  # global test_exps
  # test_exps = [str(s) for s in list(test_df['Exp'])]
  
  print 'Done'
  return