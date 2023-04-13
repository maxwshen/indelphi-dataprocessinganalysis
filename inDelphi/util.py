from mylib import util
import pickle, subprocess, os
from dataclasses import dataclass
import random
import math

@dataclass
class Filenames:
    out_dir: str
    out_letters: str
    out_dir_params: str
    log_fn: str

def print_and_log(text: str, log_fn: str):
    with open(log_fn, 'a') as f:
        f.write(text + '\n')
    print(text)
    return

def get_data(data_url: str, log_fn: str):
    print_and_log("Loading data...", log_fn)
    inp_dir = '../pickle_data/'
    return pickle.load(open(inp_dir + data_url, 'rb'))

def alphabetize(num):
    assert num < 26 ** 3, 'num bigger than 17576'
    mapper = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
              13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
              24: 'y', 25: 'z'}
    hundreds = int(num / (26 * 26)) % 26
    tens = int(num / 26) % 26
    ones = num % 26
    return ''.join([mapper[hundreds], mapper[tens], mapper[ones]])

def copy_script(out_dir):
    src_dir = '/cluster/mshen/prj/mmej_figures/src/'
    script_nm = __file__
    subprocess.call('cp ' + src_dir + script_nm + ' ' + out_dir, shell=True)
    return
def count_num_folders(out_dir):
    for fold in os.listdir(out_dir):
        assert os.path.isdir(out_dir + fold), 'Not a folder!'
    return len(os.listdir(out_dir))

def init_folders(out_place):
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

    return out_dir, out_letters, out_dir_params, log_fn

def split_data_set(master_data):
    counts_data_set = master_data['counts']
    del_features_data_set = master_data['del_features']

    all_samples = counts_data_set.index.unique(level="Sample_Name").values
    k = math.floor(len(all_samples) * 0.8)
    training_samples = random.sample(list(all_samples), k)
    counts_training_data = counts_data_set.loc[training_samples]
    counts_test_data = counts_data_set.drop(training_samples)
    del_features_training_data = del_features_data_set.loc[training_samples]
    del_features_test_data = del_features_data_set.drop(training_samples)

    training_set = {'counts': counts_training_data,  'del_features': del_features_training_data}
    test_set = {'counts': counts_test_data, 'del_features': del_features_test_data}

    return training_set, test_set