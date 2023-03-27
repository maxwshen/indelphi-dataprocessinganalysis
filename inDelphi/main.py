import helpers.neural_network as nn
import helpers.nearest_neighbours as knn

def train_model(data_url, out_place):
    nn_params, nn_2_params = nn.create(data_url, out_place)
    rate_model, bp_model, normalizer = knn.train()

if __name__ == '__main__':
    train_model('../pickle_data/inDelphi_counts_and_deletion_features.pkl', './cluster/mshen/prj/mmej_figures/out/d2_model/')