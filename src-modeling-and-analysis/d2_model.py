# Model including a neural net in autograd

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import multigrad, grad
from autograd.util import flatten
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from collections import defaultdict
import sys, string, pickle, subprocess, os, datetime
from mylib import util
import seaborn as sns, pandas as pd
from matplotlib.colors import Normalize
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib.patches as mpatches
NAME = util.get_fn(__file__)

### Define neural network ###
def relu(x):       return np.maximum(0, x)
def sigmoid(x):    return 0.5 * (np.tanh(x) + 1.0)
def logsigmoid(x): return x - np.logaddexp(0, x)
def leaky_relu(x): return np.maximum(0, x) + np.minimum(0, x) * 0.001

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
  """Build a list of (weights, biases) tuples,
     one for each layer in the net."""
  return [(scale * rs.randn(m, n),   # weight matrix
           scale * rs.randn(n))      # bias vector
          for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def batch_normalize(activations):
  mbmean = np.mean(activations, axis=0, keepdims=True)
  return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def nn_match_score_function(params, inputs):
  # """Params is a list of (weights, bias) tuples.
  #    inputs is an (N x D) matrix."""
  inpW, inpb = params[0]
  # inputs = swish(np.dot(inputs, inpW) + inpb)
  inputs = sigmoid(np.dot(inputs, inpW) + inpb)
  # inputs = leaky_relu(np.dot(inputs, inpW) + inpb)
  for W, b in params[1:-1]:
    outputs = np.dot(inputs, W) + b
    # inputs = swish(outputs)
    inputs = sigmoid(outputs)
    # inputs = logsigmoid(outputs)
    # inputs = leaky_relu(outputs)
  outW, outb = params[-1]
  outputs = np.dot(inputs, outW) + outb
  return outputs.flatten()

##
# Objective
##
def main_objective(nn_params, nn2_params, inp, obs, obs2, del_lens, num_samples, rs):
  LOSS = 0
  for idx in range(len(inp)):

    ##
    # MH-based deletion frequencies
    ##
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25*Js)
    
    # Add MH-less contribution at full MH deletion lengths
    mh_vector = inp[idx].T[0]
    mhfull_contribution = np.zeros(mh_vector.shape)
    for jdx in range(len(mh_vector)):
      if del_lens[idx][jdx] == mh_vector[jdx]:
        dl = del_lens[idx][jdx]
        mhless_score = nn_match_score_function(nn2_params, np.array(dl))
        mhless_score = np.exp(mhless_score - 0.25*dl)
        mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
        mhfull_contribution = mhfull_contribution + mask
    unnormalized_fq = unnormalized_fq + mhfull_contribution
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

    # Pearson correlation squared loss
    x_mean = np.mean(normalized_fq)
    y_mean = np.mean(obs[idx])
    pearson_numerator = np.sum((normalized_fq - x_mean)*(obs[idx] - y_mean))
    pearson_denom_x = np.sqrt(np.sum((normalized_fq - x_mean)**2))
    pearson_denom_y = np.sqrt(np.sum((obs[idx] - y_mean)**2))
    pearson_denom = pearson_denom_x * pearson_denom_y
    rsq = (pearson_numerator/pearson_denom)**2
    neg_rsq = rsq * -1
    LOSS += neg_rsq

    #
    # I want to make sure nn2 never outputs anything negative. 
    # Sanity check during training.
    #

    ##
    # Deletion length frequencies, only up to 28
    #   (Restricts training to library data, else 27 bp.)
    ##
    dls = np.arange(1, 28+1)
    dls = dls.reshape(28, 1)
    nn2_scores = nn_match_score_function(nn2_params, dls)
    unnormalized_nn2 = np.exp(nn2_scores - 0.25*np.arange(1, 28+1))

    # iterate through del_lens vector, adding mh_scores (already computed above) to the correct index
    mh_contribution = np.zeros(28,)
    for jdx in range(len(Js)):
      dl = Js[jdx]
      if dl > 28:
        break
      mhs = np.exp(mh_scores[jdx] - 0.25*dl)
      mask = np.concatenate([np.zeros(dl - 1,), np.ones(1, ) * mhs, np.zeros(28 - (dl - 1) - 1,)])
      mh_contribution = mh_contribution + mask
    unnormalized_nn2 = unnormalized_nn2 + mh_contribution
    normalized_fq = np.divide(unnormalized_nn2, np.sum(unnormalized_nn2))

    # Pearson correlation squared loss
    x_mean = np.mean(normalized_fq)
    y_mean = np.mean(obs2[idx])
    pearson_numerator = np.sum((normalized_fq - x_mean)*(obs2[idx] - y_mean))
    pearson_denom_x = np.sqrt(np.sum((normalized_fq - x_mean)**2))
    pearson_denom_y = np.sqrt(np.sum((obs2[idx] - y_mean)**2))
    pearson_denom = pearson_denom_x * pearson_denom_y
    rsq = (pearson_numerator/pearson_denom)**2
    neg_rsq = rsq * -1
    LOSS += neg_rsq

    # L2-Loss
    # LOSS += np.sum((normalized_fq - obs[idx])**2)
  return LOSS / num_samples

##
# Regularization 
##


##
# ADAM Optimizer
##
def exponential_decay(step_size):
  if step_size > 0.001:
      step_size *= 0.999
  return step_size

def adam_minmin(grad_both, init_params_nn, init_params_nn2, callback=None, num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
  x_nn, unflatten_nn = flatten(init_params_nn)
  x_nn2, unflatten_nn2 = flatten(init_params_nn2)

  m_nn, v_nn = np.zeros(len(x_nn)), np.zeros(len(x_nn))
  m_nn2, v_nn2 = np.zeros(len(x_nn2)), np.zeros(len(x_nn2))
  for i in range(num_iters):
    g_nn_uf, g_nn2_uf = grad_both(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)
    g_nn, _ = flatten(g_nn_uf)
    g_nn2, _ = flatten(g_nn2_uf)

    if callback: 
      callback(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)
    
    step_size = exponential_decay(step_size)

    # Update parameters
    m_nn = (1 - b1) * g_nn      + b1 * m_nn  # First  moment estimate.
    v_nn = (1 - b2) * (g_nn**2) + b2 * v_nn  # Second moment estimate.
    mhat_nn = m_nn / (1 - b1**(i + 1))    # Bias correction.
    vhat_nn = v_nn / (1 - b2**(i + 1))
    x_nn = x_nn - step_size * mhat_nn / (np.sqrt(vhat_nn) + eps)

    # Update parameters
    m_nn2 = (1 - b1) * g_nn2      + b1 * m_nn2  # First  moment estimate.
    v_nn2 = (1 - b2) * (g_nn2**2) + b2 * v_nn2  # Second moment estimate.
    mhat_nn2 = m_nn2 / (1 - b1**(i + 1))    # Bias correction.
    vhat_nn2 = v_nn2 / (1 - b2**(i + 1))
    x_nn2 = x_nn2 - step_size * mhat_nn2 / (np.sqrt(vhat_nn2) + eps)
  return unflatten_nn(x_nn), unflatten_nn2(x_nn2)


##
# Setup environment
##
def alphabetize(num):
  assert num < 26**3, 'num bigger than 17576'
  mapper = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
  hundreds = int(num / (26*26)) % 26
  tens = int(num / 26) % 26
  ones = num % 26
  return ''.join([mapper[hundreds], mapper[tens], mapper[ones]])

def count_num_folders(out_dir):
  for fold in os.listdir(out_dir):
    assert os.path.isdir(out_dir + fold), 'Not a folder!'
  return len(os.listdir(out_dir))

def copy_script(out_dir):
  src_dir = '/cluster/mshen/prj/mmej_figures/src/'
  script_nm = __file__
  subprocess.call('cp ' + src_dir + script_nm + ' ' + out_dir, shell = True)
  return

def print_and_log(text, log_fn):
  with open(log_fn, 'a') as f:
    f.write(text + '\n')
  print(text)
  return

##
# Plotting and Writing
##
def save_parameters(nn_params, nn2_params, out_dir_params, letters):
  pickle.dump(nn_params, open(out_dir_params + letters + '_nn.pkl', 'w'))
  pickle.dump(nn2_params, open(out_dir_params + letters + '_nn2.pkl', 'w'))
  return

def rsq(nn_params, nn2_params, inp, obs, obs2, del_lens, num_samples, rs):
  rsqs1, rsqs2 = [], []
  for idx in range(len(inp)):
    ##
    # MH-based deletion frequencies
    ##
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25*Js)
    
    # Add MH-less contribution at full MH deletion lengths
    mh_vector = inp[idx].T[0]
    mhfull_contribution = np.zeros(mh_vector.shape)
    for jdx in range(len(mh_vector)):
      if del_lens[idx][jdx] == mh_vector[jdx]:
        dl = del_lens[idx][jdx]
        mhless_score = nn_match_score_function(nn2_params, np.array(dl))
        mhless_score = np.exp(mhless_score - 0.25*dl)
        mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
        mhfull_contribution = mhfull_contribution + mask
    unnormalized_fq = unnormalized_fq + mhfull_contribution
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

    rsq1 = pearsonr(normalized_fq, obs[idx])[0]**2
    rsqs1.append(rsq1)

    ##
    # Deletion length frequencies, only up to 28
    #   (Restricts training to library data, else 27 bp.)
    ##
    dls = np.arange(1, 28+1)
    dls = dls.reshape(28, 1)
    nn2_scores = nn_match_score_function(nn2_params, dls)
    unnormalized_nn2 = np.exp(nn2_scores - 0.25*np.arange(1, 28+1))

    # iterate through del_lens vector, adding mh_scores (already computed above) to the correct index
    mh_contribution = np.zeros(28,)
    for jdx in range(len(Js)):
      dl = Js[jdx]
      if dl > 28:
        break
      mhs = np.exp(mh_scores[jdx] - 0.25*dl)
      mask = np.concatenate([np.zeros(dl - 1,), np.ones(1, ) * mhs, np.zeros(28 - (dl - 1) - 1,)])
      mh_contribution = mh_contribution + mask
    unnormalized_nn2 = unnormalized_nn2 + mh_contribution
    normalized_fq = np.divide(unnormalized_nn2, np.sum(unnormalized_nn2))


    rsq2 = pearsonr(normalized_fq, obs2[idx])[0]**2
    rsqs2.append(rsq2)

  return rsqs1, rsqs2

def save_rsq_params_csv(nms, rsqs, nn2_params, out_dir, iter_nm, data_type):
  with open(out_dir + iter_nm + '_' + data_type + '_rsqs_params.csv', 'w') as f:
    f.write( ','.join(['Exp', 'Rsq']) + '\n')
    for i in xrange(len(nms)):
      f.write( ','.join([nms[i], str(rsqs[i])]) + '\n' )
  return

def save_train_test_names(train_nms, test_nms, out_dir):
  with open(out_dir + 'train_exps.csv', 'w') as f:
    f.write( ','.join(['Exp']) + '\n')
    for i in xrange(len(train_nms)):
      f.write( ','.join([train_nms[i]]) + '\n' )
  with open(out_dir + 'test_exps.csv', 'w') as f:
    f.write( ','.join(['Exp']) + '\n')
    for i in xrange(len(test_nms)):
      f.write( ','.join([test_nms[i]]) + '\n' )
  return

def plot_mh_score_function(nn_params, out_dir, letters):
  data = defaultdict(list)
  col_names = ['MH Length', 'GC', 'MH Score']
  # Add normal MH
  for ns in range(5000):
    length = np.random.choice(range(1, 28+1))
    gc = np.random.uniform()
    features = np.array([length, gc])
    ms = nn_match_score_function(nn_params, features)[0]
    data['Length'].append(length)
    data['GC'].append(gc)
    data['MH Score'].append(ms)
  df = pd.DataFrame(data)

  with PdfPages(out_dir + letters + '_matchfunction.pdf', 'w') as pdf:
    # Plot length vs. match score
    sns.violinplot(x = 'Length', y = 'MH Score', data = df, scale = 'width')
    plt.title('Learned Match Function: MH Length vs. MH Score')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot GC vs match score, color by length
    palette = sns.color_palette('hls', max(df['Length']) + 1)
    for length in range(1, max(df['Length'])+1):
      ax = sns.regplot(x = 'GC', y = 'MH Score', data = df.loc[df['Length']==length], color = palette[length-1], label = 'Length: %s' % (length))
    plt.legend(loc = 'best')
    plt.xlim([0, 1])
    plt.title('GC vs. MH Score, colored by MH Length')
    pdf.savefig()
    plt.close()
  return

def plot_pred_obs(nn_params, nn2_params, inp, obs, del_lens, nms, datatype, letters):
  num_samples = len(inp)
  [beta] = nn2_params
  pred = []
  obs_dls = []
  for idx in range(len(inp)):
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - beta*Js)
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))
    curr_pred = np.zeros(28 - 1 + 1)
    curr_obs = np.zeros(28 - 1 + 1)
    for jdx in range(len(del_lens[idx])):
      dl_idx = int(del_lens[idx][jdx]) - 1
      curr_pred[dl_idx] += normalized_fq[jdx]
      curr_obs[dl_idx] += obs[idx][jdx]
    pred.append(curr_pred.flatten())
    obs_dls.append(curr_obs.flatten())

  ctr = 0
  with PdfPages(out_dir + letters + '_' + datatype + '.pdf', 'w') as pdf:
    for idx in range(num_samples):
      ymax = max(max(pred[idx]), max(obs_dls[idx])) + 0.05
      rsq = pearsonr(obs_dls[idx], pred[idx])[0]**2

      plt.subplot(211)
      plt.title('Designed Oligo %s, Rsq=%s' % (nms[idx], rsq))
      plt.bar(range(1, 28+1), obs_dls[idx], align = 'center', color = '#D00000')
      plt.xlim([0, 28+1])
      plt.ylim([0, ymax])
      plt.ylabel('Observed')

      plt.subplot(212)
      plt.bar(range(1, 28+1), pred[idx], align = 'center', color = '#FFBA08')
      plt.xlim([0, 26+1])
      plt.ylim([0, ymax])
      plt.ylabel('Predicted')

      pdf.savefig()
      plt.close()
      ctr += 1
      if ctr >= 50:
        break
  return



##
# Setup / Run Main
##
if __name__ == '__main__':
  out_place = '/cluster/mshen/prj/mmej_figures/out/%s/' % (NAME)
  util.ensure_dir_exists(out_place)
  num_folds = count_num_folders(out_place)
  out_letters = alphabetize(num_folds + 1)
  out_dir = out_place + out_letters + '/'
  out_dir_params = out_place + out_letters + '/parameters/'
  util.ensure_dir_exists(out_dir)
  copy_script(out_dir)
  util.ensure_dir_exists(out_dir_params)

  log_fn = out_dir + '_log_%s.out' % (out_letters)
  with open(log_fn, 'w') as f:
    pass
  print_and_log('out dir: ' + out_letters, log_fn)

  counter = 0
  seed = npr.RandomState(1)

  '''
  Model hyper-parameters
  '''
  nn_layer_sizes = [2, 16, 16, 1]
  nn2_layer_sizes = [1, 16, 16, 1]

  print_and_log("Loading data...", log_fn)
  inp_dir = '/cluster/mshen/prj/mmej_figures/out/c2_model_dataset/'
  # master_data = pickle.load(open(inp_dir + 'dataset_try1.pkl'))
  # master_data = pickle.load(open(inp_dir + 'dataset_try2.pkl'))
  # master_data = pickle.load(open(inp_dir + 'dataset_try3.pkl'))
  master_data = pickle.load(open(inp_dir + 'dataset_try4.pkl'))

  '''
  Unpack data from e11_dataset
  '''
  [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = master_data
  INP = []
  for mhl, gcf in zip(mh_lens, gc_fracs):
    inp_point = np.array([mhl, gcf]).T   # N * 2
    INP.append(inp_point)
  INP = np.array(INP)   # 2000 * N * 2
  # Neural network considers each N * 2 input, transforming it into N * 1 output.
  OBS = np.array(freqs)
  OBS2 = np.array(dl_freqs)
  NAMES = np.array([str(s) for s in exps])
  DEL_LENS = np.array(del_lens)

  ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size = 0.15, random_state = seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans
  save_train_test_names(NAMES_train, NAMES_test, out_dir)


  ''' 
  Training parameters
  '''
  param_scale = 0.1
  num_epochs = 7*200 + 1
  step_size = 0.10

  init_nn_params = init_random_params(param_scale, nn_layer_sizes, rs = seed)
  # init_nn_params = pickle.load(open('/cluster/mshen/prj/mmej_manda/out/2017-08-23/i2_model_mmh/aax/parameters/aav_nn.pkl'))

  init_nn2_params = init_random_params(param_scale, nn2_layer_sizes, rs = seed)

  # batch_size = len(INP_train)   # use all of training data
  batch_size = 200
  num_batches = int(np.ceil(len(INP_train) / batch_size))
  def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

  def objective(nn_params, nn2_params, iter):
    idx = batch_indices(iter)
    return main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)

  both_objective_grad = multigrad(objective, argnums=[0,1])

  def print_perf(nn_params, nn2_params, iter):
    print_and_log(str(iter), log_fn)
    if iter % 5 != 0:
      return None
    
    train_loss = main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
    test_loss = main_objective(nn_params, nn2_params, INP_test, OBS_test, OBS2_train, DEL_LENS_test, len(INP_test), seed)

    tr1_rsq, tr2_rsq = rsq(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
    te1_rsq, te2_rsq = rsq(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test, len(INP_test), seed)
    
    out_line = ' %s  | %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t|' % (iter, train_loss, np.mean(tr1_rsq), np.mean(tr2_rsq), test_loss, np.mean(te1_rsq), np.mean(te2_rsq))
    print_and_log(out_line, log_fn)

    if iter % 20 == 0:
      letters = alphabetize(int(iter/10))
      print_and_log(" Iter | Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2", log_fn)
      print_and_log('%s %s %s' % (datetime.datetime.now(), out_letters, letters), log_fn)
      save_parameters(nn_params, nn2_params, out_dir_params, letters)
      # save_rsq_params_csv(NAMES_test, test_rsqs, nn2_params, out_dir, letters, 'test')
      if iter >= 10:
      # if iter >= 0:
        pass
        # plot_mh_score_function(nn_params, out_dir, letters + '_nn')
        # plot_pred_obs(nn_params, nn2_params, INP_train, OBS_train, DEL_LENS_train, NAMES_train, 'train', letters)
        # plot_pred_obs(nn_params, nn2_params, INP_test, OBS_test, DEL_LENS_test, NAMES_test, 'test', letters)

    return None

  optimized_params = adam_minmin(both_objective_grad,
                                  init_nn_params, 
                                  init_nn2_params, 
                                  step_size = step_size, 
                                  num_iters = num_epochs,
                                  callback = print_perf)

  print('Done')
