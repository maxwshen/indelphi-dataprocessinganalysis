import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
from matplotlib.lines import Line2D


# Fit multiple known distributions against a subset
def distribution_fit(data):
    size = 100
    x = np.arange(0, size, 1)
    y = data["fraction"] * 100

    # Distributions used for data fitting
    dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']

    # Plot data
    plt.hist(y, density=True, bins=150, alpha=0.5)

    # Find the best-fit parameters for each distribution
    for dist_name in dist_names:
        dist = getattr(sc, dist_name)
        params = dist.fit(y)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        if arg:
            pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)
        else:
            pdf_fitted = dist.pdf(x, loc=loc, scale=scale)

        # PLot distributions
        plt.plot(pdf_fitted, label=dist_name)

    plt.legend(loc='upper right')
    plt.show()

# Fit Poisson distribution to two subsets
def poisson_fit(a, b, titles=["", ""]):
    # poisson distribution data
    x = np.arange(0, 100, 1)
    y_a = sc.poisson.pmf(x, mu=np.mean(a["fraction"] * 100))
    y_b = sc.poisson.pmf(x, mu=np.mean(b["fraction"] * 100))

    plt.subplot(1, 2, 1)
    plt.hist(a["fraction"] * 100, density=True, bins=150, alpha=1.0)
    plt.plot(x, y_a, c="r", linewidth=2)
    plt.title(titles[0])

    plt.subplot(1, 2, 2)
    plt.hist(b["fraction"] * 100, density=True, bins=150, alpha=1.0)
    plt.plot(x, y_b, c="r", linewidth=2)
    plt.title(titles[1])

    plt.suptitle("distributions")
    plt.tight_layout(pad=1.0)
    plt.show()


# Fit Poisson distribution to all subsets
def poisson_fit_all(data):
    i = 1
    x = np.arange(0, 100, 1)
    for subset in data:
        plt.subplot(2, -(len(data) // -2), i)
        plt.hist(data[subset]["fraction"] * 100, density=True, bins=150, alpha=1.0)
        plt.plot(x, sc.poisson.pmf(x, mu=np.mean(data[subset]["fraction"] * 100)), c="r", linewidth=1)
        plt.title(subset)
        i += 1

    plt.suptitle("distributions")
    plt.tight_layout(pad=1.0)
    plt.show()


# Fit gamma distribution to two subsets
def gamma_fit(a, b, titles=["", ""]):
    # plt.subplot(1, 2, 1)
    plt.hist(a["fraction"] * 100, density=True, color="tab:blue", bins=150, alpha=0.5)
    pdf_fitted_a = gamma(a)
    plt.plot(pdf_fitted_a, c="tab:blue", linewidth=2)
    # plt.subplot(1, 2, 2)
    plt.hist(b["fraction"] * 100, density=True, color="tab:red", bins=150, alpha=0.5)
    pdf_fitted_b = gamma(b)
    plt.plot(pdf_fitted_b, c="tab:red", linewidth=2)
    plt.ylim(0.0, 0.2)
    labels = ["MH data", "MH fit", "MH-less data", "MH-less fit"]
    legend_markers = [Line2D([0], [0], lw=4, marker="_", color="tab:blue", alpha=0.5, markersize=10),
                      Line2D([0], [0], lw=4, marker="_", color="tab:blue", markersize=10),
                      Line2D([0], [0], lw=4, marker="_", color="tab:red", alpha=0.5, markersize=10),
                      Line2D([0], [0], lw=4, marker="_", color="tab:red", markersize=10)]
    plt.legend(legend_markers, labels, loc=1)
    plt.ylabel("Normalized counts")
    plt.xlabel("1-bp deletion frequency (%)")
    plt.title("Fitting gamma distribution to subsets")
    plt.show()


# Fit gamma distribution to all subsets
def gamma_fit_all(data):
    i = 1
    for subset in data:
        plt.subplot(2, -(len(data) // -2), i)
        # plt.subplot(1, len(data), i)
        plt.hist(data[subset]["fraction"] * 100, density=True, bins=150, alpha=1.0)
        plt.plot(gamma(data[subset]), c="r", linewidth=2)
        plt.title(subset)
        plt.ylim(0, 0.2)
        i += 1
    plt.suptitle("distributions")
    plt.tight_layout(pad=1.0)
    plt.show()


# Helper function for gamma fitting
def gamma(data):
    x = np.arange(0, 100, 1)
    dist = getattr(sc, "gamma")
    params = dist.fit(data["fraction"] * 100)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    if arg:
        pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)
    else:
        pdf_fitted = dist.pdf(x, loc=loc, scale=scale)
    return pdf_fitted


# Fit normal distribution to two subsets
def normal_fit(a, b):
    # plt.subplot(1, 2, 1)
    plt.hist(a["fraction"] * 100, density=True, color="tab:blue", bins=150, alpha=0.5)
    pdf_fitted_a = normal(a)
    plt.plot(pdf_fitted_a, c="tab:blue", linewidth=2)
    # plt.subplot(1, 2, 2)
    plt.hist(b["fraction"] * 100, density=True, color="tab:red", bins=150, alpha=0.5)
    pdf_fitted_b = normal(b)
    plt.plot(pdf_fitted_b, c="tab:red", linewidth=2)
    plt.ylim(0.0, 0.2)
    labels = ["MH data", "MH fit", "MH-less data", "MH-less fit"]
    legend_markers = [Line2D([0], [0], lw=4, marker="_", color="tab:blue", alpha=0.5, markersize=10),
                      Line2D([0], [0], lw=4, marker="_", color="tab:blue", markersize=10),
                      Line2D([0], [0], lw=4, marker="_", color="tab:red", alpha=0.5, markersize=10),
                      Line2D([0], [0], lw=4, marker="_", color="tab:red", markersize=10)]
    plt.legend(legend_markers, labels, loc=1)
    plt.ylabel("Normalized counts")
    plt.xlabel("1-bp deletion frequency (%)")
    plt.title("Fitting normal distribution to subsets")
    plt.show()


# Helper function for normal fitting
def normal(data):
    x = np.arange(0, 100, 1)
    dist = getattr(sc, "norm")
    params = dist.fit(data["fraction"] * 100)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    if arg:
        pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)
    else:
        pdf_fitted = dist.pdf(x, loc=loc, scale=scale)
    return pdf_fitted


# Compute the variances of all subsets
def variances(data):
    variances = []
    for subset in data:
        variance = np.var(data[subset]["fraction"] * 100)
        variances.append(variance)
        print("Variance of " + subset + ": " + str(variance))


# Perform Welch's t-test to samples a and b
def t_test(a, b, title=""):
    print("Tested subsets: " + title)
    print(sc.ttest_ind(a["fraction"], b["fraction"], equal_var=False))
