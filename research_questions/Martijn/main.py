import scipy.stats

from plot_deletions import *
import filter_deletions as fd
import statistics as st
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = fd.filter_data("inDelphi_counts_and_deletion_features.pkl")
    all_dels = data["all_one_bp_dels"]
    mh = data["microhomology"]
    mhless = data["microhomologyless"]
    pos_0 = data["mhless_0+1"]
    pos_min_1 = data["mhless_-1+1"]
    cg_0 = data["cg_0"]
    cg_50 = data["cg_50"]
    cg_100 = data["cg_100"]

    plot_XX_XY(mh, mhless, title="Frequency of 1-bp deletions for different target contexts")
    plot_AA_XY(mh, mhless, title="Frequency of 1-bp deletions for different target contexts")
    plot_CG([cg_0, cg_50, cg_100], title="Frequency of 1-bp deletions for different CG-contents")
    plot_XX_XY(pos_0, pos_min_1, labels=["XY", "YX"], title="MH-less 1-bp deletion frequencies, where X is deleted")

    plot_distributions(data)
    st.variances(data)
    # st.distribution_fit(mh)
    st.gamma_fit(mh, mhless)
    st.normal_fit(mh, mhless)
    st.t_test(mh, mhless, "MH vs. MH-less")
    st.t_test(cg_0, cg_50, "CG-0 vs. CG-50")
    st.t_test(cg_50, cg_100, "CG-50 vs. CG-100")
    st.t_test(cg_0, cg_100, "CG-0 vs. CG-100")
    st.t_test(pos_0, pos_min_1, "-4 del vs. -3 del")

    # Later addition: TT-MH vs. AA/CC/GG-MH
    TT = mh.loc[mh["cutsite"].isin(["TT"])]
    MH = mh.loc[mh["cutsite"].isin(["AA", "CC", "GG"])]
    st.t_test(TT, MH, "TT-MH vs. other MH")
    # print(scipy.stats.ttest_ind([1,1,1,1,1,1,2,2], [1,1,1,1,2,2,2,2,1,1]))
print("Done")
