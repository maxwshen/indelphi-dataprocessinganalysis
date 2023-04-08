import inDelphi.neural_network as nn
# import inDelphi.nearest_neighbours as knn
from util import get_data, init_folders, Filenames
def train_model(data_url, out_place):
    out_dir, out_letters, out_dir_params, log_fn = init_folders(out_place)
    filenames = Filenames(out_dir, out_letters, out_dir_params, log_fn)
    master_data = get_data(data_url, log_fn)

    nn_params, nn_2_params = nn.train_and_create(master_data, filenames)
    # rate_model, bp_model, normalizer = knn.train(master_data, filenames)

if __name__ == '__main__':
    train_model('../pickle_data/inDelphi_counts_and_deletion_features.pkl', './cluster/mshen/prj/mmej_figures/out/d2_model/')