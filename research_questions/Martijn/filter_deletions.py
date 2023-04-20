import pickle

def filter_data(pickle_name):
    # Load the dataframe of repair outcomes
    data = pickle.load(open(pickle_name, "rb"))
    df = data["counts"]
    filtered_data = {}

    # Filter dataframe on 1-bp deletions around the cutsite
    # Store the nucleotides surrounding the cut
    one_bp_del_indices = []
    cutsites = []
    positions = []
    for i in range(len(df.index)):
        if df.index[i][1] == "-1+1" or df.index[i][1] == "0+1":
            if df.index[i][1] == "0+1":
                positions.append(0)
            else:
                positions.append(-1)

            one_bp_del_indices.append(df.index[i])
            cutsites.append(df.index[i][0][-4:-2])

    one_bp_dels = df.filter(items=one_bp_del_indices, axis=0)
    one_bp_dels["cutsite"] = cutsites
    one_bp_dels["position"] = positions

    microhomology = one_bp_dels.loc[one_bp_dels["cutsite"].isin(["AA", "TT", "CC", "GG"])]
    mhless = one_bp_dels.loc[~one_bp_dels["cutsite"].isin(["AA", "TT", "CC", "GG"])]

    filtered_data["all_one_bp_dels"] = one_bp_dels
    filtered_data["microhomology"] = microhomology
    filtered_data["microhomologyless"] = mhless
    filtered_data["mhless_0+1"] = mhless.loc[mhless["position"] == 0]
    filtered_data["mhless_-1+1"] = mhless.loc[mhless["position"] == -1]
    filtered_data["cg_0"] = mhless.loc[mhless["cutsite"].isin(["AT", "TA"])]
    filtered_data["cg_50"] = mhless.loc[mhless["cutsite"].isin(["AC", "AG", "TC", "TG", "CA", "GA", "CT", "GT"])]
    filtered_data["cg_100"] = mhless.loc[mhless["cutsite"].isin(["CG", "GC"])]

    return filtered_data
