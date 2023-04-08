from mylib import util
import pickle, subprocess, os, datetime
import autograd.numpy.random as npr
import pandas as pd
import autograd.numpy as np
from sklearn.model_selection import train_test_split
from autograd.differential_operators import multigrad_dict as multigrad
from autograd.misc.flatten import flatten
import tqdm

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

def exponential_decay(step_size):
    if step_size > 0.001:
        step_size *= 0.999
    return step_size

def relu(x):       return np.maximum(0, x)

def sigmoid(x):    return 0.5 * (np.tanh(x) + 1.0)

def logsigmoid(x): return x - np.logaddexp(0, x)

def leaky_relu(x): return np.maximum(0, x) + np.minimum(0, x) * 0.001