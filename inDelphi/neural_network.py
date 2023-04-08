from mylib import util
import pickle, subprocess, os, datetime
import autograd.numpy.random as npr
import pandas as pd
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from autograd.differential_operators import multigrad_dict as multigrad
from autograd.misc.flatten import flatten
from util import print_and_log, get_data, alphabetize, Filenames
import tqdm

def save_train_test_names(train_nms, test_nms, out_dir):
    with open(out_dir + 'train_exps.csv', 'w') as f:
        f.write(','.join(['Exp']) + '\n')
        for i in range(len(train_nms)):
            f.write(','.join([train_nms[i]]) + '\n')
    with open(out_dir + 'test_exps.csv', 'w') as f:
        f.write(','.join(['Exp']) + '\n')
        for i in range(len(test_nms)):
            f.write(','.join([test_nms[i]]) + '\n')
    return

def exponential_decay(step_size):
    if step_size > 0.001:
        step_size *= 0.999
    return step_size

def relu(x):       return np.maximum(0, x)

def sigmoid(x):    return 0.5 * (np.tanh(x) + 1.0)

def logsigmoid(x): return x - np.logaddexp(0, x)

def leaky_relu(x): return np.maximum(0, x) + np.minimum(0, x) * 0.001


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
     one for each layer in the net."""
    return [(scale * rs.randn(m, n),  # weight matrix
             scale * rs.randn(n))  # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_nn():
    nn_layer_sizes = [2, 16, 16, 1]
    nn2_layer_sizes = [1, 16, 16, 1]

    return nn_layer_sizes, nn2_layer_sizes

def unpack_data(master_data):
    res = pd.merge(master_data['counts'], master_data['del_features'], left_on=master_data['counts'].index,
                   right_on=master_data['del_features'].index)
    res[['sample', 'offset']] = pd.DataFrame(res['key_0'].tolist(), index=res.index)
    mh_lens = []
    gc_fracs = []
    del_lens = []
    exps = []
    freqs = []
    dl_freqs = []
    for group in res.groupby("sample"):
        mh_lens.append(group[1]['homologyLength'].values)
        gc_fracs.append(group[1]['homologyGCContent'].values)
        del_lens.append(group[1]['Size'].values)
        exps.append(group[1]['key_0'].values)
        freqs.append(group[1]['countEvents'].values)
        dl_freqs.append(group[1]['fraction'].values)

    return mh_lens, gc_fracs, del_lens, exps, freqs, dl_freqs

def create_test_set(INP, OBS, OBS2, NAMES, DEL_LENS, out_dir, seed):
    ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size=0.15, random_state=seed)
    INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans
    save_train_test_names(NAMES_train, NAMES_test, out_dir)
    return INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test
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

def save_parameters(nn_params, nn2_params, out_dir_params, letters):
    pickle.dump(nn_params, open(out_dir_params + letters + '_nn.pkl', 'wb'))
    pickle.dump(nn2_params, open(out_dir_params + letters + '_nn2.pkl', 'wb'))
    return

def adam_minmin(grad_both, init_params_nn, init_params_nn2, callback=None, num_iters=100, step_size=0.001, b1=0.9,
                b2=0.999, eps=10 ** -8):
    x_nn, unflatten_nn = flatten(init_params_nn)
    x_nn2, unflatten_nn2 = flatten(init_params_nn2)

    m_nn, v_nn = np.zeros(len(x_nn)), np.zeros(len(x_nn))
    m_nn2, v_nn2 = np.zeros(len(x_nn2)), np.zeros(len(x_nn2))
    for i in range(num_iters):
        print(f"Optimization iteration {i}")
        output = grad_both(unflatten_nn(x_nn), unflatten_nn2(x_nn2))
        g_nn_uf, g_nn2_uf = (output["nn_params"], output["nn2_params"])
        g_nn, _ = flatten(g_nn_uf)
        g_nn2, _ = flatten(g_nn2_uf)

        if callback:
            callback(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)

        step_size = exponential_decay(step_size)

        # Update parameters
        m_nn = (1 - b1) * g_nn + b1 * m_nn  # First  moment estimate.
        v_nn = (1 - b2) * (g_nn ** 2) + b2 * v_nn  # Second moment estimate.
        mhat_nn = m_nn / (1 - b1 ** (i + 1))  # Bias correction.
        vhat_nn = v_nn / (1 - b2 ** (i + 1))
        x_nn = x_nn - step_size * mhat_nn / (np.sqrt(vhat_nn) + eps)

        # Update parameters
        m_nn2 = (1 - b1) * g_nn2 + b1 * m_nn2  # First  moment estimate.
        v_nn2 = (1 - b2) * (g_nn2 ** 2) + b2 * v_nn2  # Second moment estimate.
        mhat_nn2 = m_nn2 / (1 - b1 ** (i + 1))  # Bias correction.
        vhat_nn2 = v_nn2 / (1 - b2 ** (i + 1))
        x_nn2 = x_nn2 - step_size * mhat_nn2 / (np.sqrt(vhat_nn2) + eps)
    return unflatten_nn(x_nn), unflatten_nn2(x_nn2)

def main_objective(nn_params, nn2_params, mh_NN_inp, obs_freqs, obs_frac, del_lens, num_samples):
    LOSS = 0
    for idx in tqdm.tqdm(range(len(mh_NN_inp)), desc="#GRNA's"):

        ##
        # MH-based deletion frequencies
        ##
        mh_scores = nn_match_score_function(nn_params, mh_NN_inp[idx])
        Js = np.array(del_lens[idx])
        unnormalized_fq = np.exp(mh_scores - 0.25 * Js)

        # Add MH-less contribution at full MH deletion lengths
        mh_vector = mh_NN_inp[idx].T[0]
        mhfull_contribution = np.zeros(mh_vector.shape)
        for jdx in range(len(mh_vector)):
            if del_lens[idx][jdx] == mh_vector[jdx]:
                dl = del_lens[idx][jdx]
                mhless_score = nn_match_score_function(nn2_params, np.array(dl))
                mhless_score = np.exp(mhless_score - 0.25 * dl)
                mask = np.concatenate(
                    [np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
                mhfull_contribution = mhfull_contribution + mask
        unnormalized_fq = unnormalized_fq + mhfull_contribution
        normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

        # Pearson correlation squared loss
        x_mean = np.mean(normalized_fq)
        y_mean = np.mean(obs_freqs[idx])
        pearson_numerator = np.sum((normalized_fq - x_mean) * (obs_freqs[idx] - y_mean))
        pearson_denom_x = np.sqrt(np.sum((normalized_fq - x_mean) ** 2))
        pearson_denom_y = np.sqrt(np.sum((obs_freqs[idx] - y_mean) ** 2))
        pearson_denom = pearson_denom_x * pearson_denom_y
        rsq = (pearson_numerator / pearson_denom) ** 2
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
        dls = np.arange(1, 28 + 1)
        dls = dls.reshape(28, 1)
        nn2_scores = nn_match_score_function(nn2_params, dls)
        unnormalized_nn2 = np.exp(nn2_scores - 0.25 * np.arange(1, 28 + 1))

        # iterate through del_lens vector, adding mh_scores (already computed above) to the correct index
        mh_contribution = np.zeros(28, )
        for jdx in range(len(Js)):
            dl = Js[jdx]
            if dl > 28:
                break
            mhs = np.exp(mh_scores[jdx] - 0.25 * dl)
            mask = np.concatenate([np.zeros(dl - 1, ), np.ones(1, ) * mhs, np.zeros(28 - (dl - 1) - 1, )])
            mh_contribution = mh_contribution + mask
        unnormalized_nn2 = unnormalized_nn2 + mh_contribution
        normalized_fq = np.divide(unnormalized_nn2, np.sum(unnormalized_nn2))

        # Pearson correlation squared loss
        x_mean = np.mean(normalized_fq)
        y_mean = np.mean(obs_frac[idx])
        pearson_numerator = np.sum((normalized_fq - x_mean) * (obs_frac[idx][0:28] - y_mean))
        pearson_denom_x = np.sqrt(np.sum((normalized_fq - x_mean) ** 2))
        pearson_denom_y = np.sqrt(np.sum((obs_frac[idx] - y_mean) ** 2))
        pearson_denom = pearson_denom_x * pearson_denom_y
        rsq = (pearson_numerator / pearson_denom) ** 2
        neg_rsq = rsq * -1
        LOSS += neg_rsq

        # L2-Loss
        # LOSS += np.sum((normalized_fq - obs[idx])**2)
    return LOSS / num_samples

def init_training_param(nn_layer_sizes, nn2_layer_sizes, seed, INP_train, param_scale, batch_size):
    init_nn_params = init_random_params(param_scale, nn_layer_sizes, rs=seed)
    init_nn2_params = init_random_params(param_scale, nn2_layer_sizes, rs=seed)

    num_batches = int(np.ceil(len(INP_train) / batch_size))

    return init_nn_params, init_nn2_params, num_batches

def init_model(seed, INP_train, param_scale, batch_size):
    nn_layer_sizes, nn2_layer_sizes = init_nn()
    return init_training_param(nn_layer_sizes,nn2_layer_sizes, seed=seed,
                               INP_train=INP_train,param_scale=param_scale,batch_size=batch_size)

def init_test_set(master_data, filenames, seed):
    mh_lens, gc_fracs, del_lens, exps, freqs, dl_freqs = unpack_data(master_data)

    INP = []
    for mhl, gcf in zip(mh_lens, gc_fracs):
        inp_point = np.array([mhl, gcf]).T  # N * 2
        INP.append(inp_point)
    INP = np.array(INP, dtype=object)  # 2000 * N * 2

    OBS_FREQS = np.array(freqs, dtype=object)
    OBS_FRAC = np.array(dl_freqs, dtype=object)
    NAMES = np.array([str(s) for s in exps])
    DEL_LENS = np.array(del_lens, dtype=object)
    return create_test_set(INP, OBS_FREQS, OBS_FRAC, NAMES, DEL_LENS, filenames.out_dir, seed)

def train_and_create(master_data: dict, filenames: Filenames,
                     num_epochs=10, param_scale = 0.1, step_size = 0.10, batch_size=200):
    """
    Trains and creates the MH-NN and MH-less NN
    """
    seed = npr.RandomState(1)

    INP_train, INP_test, OBS_FREQS_train, OBS_FREQS_test, OBS_FRAC_train, OBS_FRAC_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = init_test_set(master_data, filenames, seed)
    init_nn_params, init_nn2_params, num_batches = init_model(seed, INP_train, param_scale, batch_size)

    def objective(nn_params, nn2_params):
        return main_objective(nn_params, nn2_params, INP_train, OBS_FREQS_train, OBS_FRAC_train, DEL_LENS_train, batch_size)

    both_objective_grad = multigrad(objective)

    def print_perf(nn_params, nn2_params, iter):
        print("= finished iteration")
        print_and_log(str(iter), filenames.log_fn)
        if iter % 5 != 0:
            return None

        train_loss = main_objective(nn_params, nn2_params, INP_train, OBS_FREQS_train, OBS_FRAC_train, DEL_LENS_train, batch_size)
        test_loss = main_objective(nn_params, nn2_params, INP_test, OBS_FREQS_test, OBS_FRAC_train, DEL_LENS_test, len(INP_test))

        # TODO RSQ broken, should be fixed
        # tr1_rsq, tr2_rsq = rsq(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size,
        #                        seed)
        # te1_rsq, te2_rsq = rsq(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test, len(INP_test))

        out_line = f"Iteration: {iter}, Train Loss: {train_loss}, Test loss: {test_loss}."
        print_and_log(out_line, filenames.log_fn)

        if iter % 20 == 0:
            letters = alphabetize(int(iter / 10))
            print_and_log(" Iter | Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2",
                          filenames.log_fn)
            print_and_log('%s %s %s' % (datetime.datetime.now(), filenames.out_letters, letters), filenames.log_fn)
            save_parameters(nn_params, nn2_params, filenames.out_dir_params, letters)
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
                                   step_size=step_size,
                                   num_iters=num_epochs,
                                   callback=print_perf)

    return optimized_params
