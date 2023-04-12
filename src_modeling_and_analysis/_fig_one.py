import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import _predict as predict
import pandas as pd
import pickle as pkl


def use_fraction(master_data, exps):
    fractions = master_data['counts']['fraction']
    result = {}

    for i, sequence in enumerate(exps):
        indels = fractions[sequence]
        print(sequence)
        for pos_indel in indels.index:
            pos, indel = pos_indel.split("+")
            if indel.isdigit():
                if int(indel) % 3 != 0:
                    # frameshift
                    fraction = fractions[sequence][pos_indel]
                    if fraction != 0.0:
                        if sequence in result:
                            result[sequence] += fraction
                        else:
                            result[sequence] = fraction
            else:
                fraction = fractions[sequence][pos_indel]
                if fraction != 0.0:
                    if sequence in result:
                        result[sequence] += fraction
                    else:
                        result[sequence] = fraction

        # only plot some datapoints for debugging
        # if i == 80000:
        #     break

    return result


def get_predicted(dataset):
    predict.init_model()
    rate_model, bp_model, normalizer = predict.init_rate_bp_models()
    dataset = pd.merge(dataset['counts'], dataset['del_features'],
                       left_on=dataset['counts'].index, right_on=dataset['del_features'].index,
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

    exps = exps[0:80]  # select fewer exps for testing purposes
    result = {}

    for i, exp in enumerate(exps):
        fraction = 0
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
            pred_del_df, pred_all_df, total_phi_score, rate_1bpins = predict.predict_all(sequence, cutsite, rate_model, bp_model, normalizer)
            for index, row in pred_all_df.iterrows():
                if row['Length'] % 3 != 0:
                    fraction += row['Predicted_Frequency']

            break # only use one cutsite for now,  still very unclear

        result[exp] = fraction
    return result, exps




if __name__ == "__main__":
    master_data = pkl.load(open('../pickle_data/inDelphi_counts_and_deletion_features_p4.pkl', 'rb'))
    predicted, exps = get_predicted(master_data)

    fraction = use_fraction(master_data, exps)

    truth = 1 - np.array(list(fraction.values()))
    predicted = 1 - np.array(list(predicted.values()))

    plt.plot([0, 1], linestyle='dashed', c='black')
    plt.scatter(truth, predicted)
    sns.regplot(x=truth, y=predicted, scatter=False, color='red')

    plt.savefig("figures/figure_one.png")

    corr = np.corrcoef(truth, predicted)
    print(corr)