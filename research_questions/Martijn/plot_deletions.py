import matplotlib.pyplot as plt
import random


# Two subset jitterplot: MH and MH-less
def plot_XX_XY(mh, mhless, labels=["XX", "XY"], title=""):
    # Constants for visualization
    x_ticks = [20, 45]
    cutsite_labels = labels
    colors = ["tab:red", "tab:blue"]

    # Add x-position, for plotting based on cutsite context, to dataframe
    x_pos = []
    color = []
    for i in range(len(mh)):
        x_pos.append(x_ticks[0] + 2 * (random.random() - 0.5))
        color.append(colors[0])
    mh["x_pos"] = x_pos
    mh["color"] = color

    # Repeat for second subset
    x_pos = []
    color = []
    for i in range(len(mhless)):
        x_pos.append(x_ticks[-1] + 2 * (random.random() - 0.5))
        color.append(colors[-1])
    mhless["x_pos"] = x_pos
    mhless["color"] = color

    # Plot datapoints
    jitterplot_data([mh, mhless], x_ticks, cutsite_labels, title)


# Jitterplot of all different MHs against MH-less
def plot_AA_XY(mh, mhless, title=""):
    # Constants for visualization
    x_ticks = [20, 45, 70, 95, 120]
    cutsite_labels = ["AA", "TT", "CC", "GG", "XY"]
    colors = ["tab:red", "tab:purple", "gold", "tab:green", "tab:blue"]

    # Add x-position and color, for plotting based on cutsite context, to dataframe
    x_pos = []
    color = []
    for i in range(len(mh)):
        x_pos.append(x_ticks[cutsite_labels.index(mh["cutsite"][i])] + 2 * (random.random() - 0.5))
        color.append(colors[cutsite_labels.index(mh["cutsite"][i])])
    mh["x_pos"] = x_pos
    mh["color"] = color

    # Repeat for second subset
    x_pos = []
    color = []
    for i in range(len(mhless)):
        x_pos.append(x_ticks[cutsite_labels.index(cutsite_labels[-1])] + 2 * (random.random() - 0.5))
        color.append(colors[-1])
    mhless["x_pos"] = x_pos
    mhless["color"] = color

    # Plot datapoints
    jitterplot_data([mh, mhless], x_ticks, cutsite_labels, title)


# Jitterplot of all different MHs against MH-less
def plot_CG(data, labels=["0", "50", "100"], title=""):
    # Constants for visualization
    x_ticks = [20, 45, 70]
    cutsite_labels = labels
    colors = ["tab:red", "tab:green", "tab:blue"]

    # Add x-position, for plotting based on cutsite context, to dataframe
    for i in range(len(data)):
        x_pos = []
        color = []
        for j in range(len(data[i])):
            x_pos.append(x_ticks[i] + 2 * (random.random() - 0.5))
            color.append(colors[i])
        data[i]["x_pos"] = x_pos
        data[i]["color"] = color

    # Plot datapoints
    jitterplot_data(data, x_ticks, cutsite_labels, title, xlabel="CG-content in context (%)")


# Helper function for creating jitterplot
def jitterplot_data(data, ticks, labels, title="", xlabel="Nucleotides at -4 and -3 position"):
    # Plot datapoints
    for i in range(len(data)):
        plt.scatter(data[i]["x_pos"], data[i]["fraction"] * 100, c=data[i]["color"], s=5.0, alpha=0.2)

    plt.xticks(ticks, labels)
    plt.xlim(left=ticks[0] - 10, right=ticks[-1] + 10)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency of 1-bp deletions, lib-A mESCs (%)")
    plt.title(title)
    plt.show()


# Plot the distribution of frequency counts per gRNA of each subset
def all_distributions(data):
    i = 1
    for subset in data:
        plt.subplot(2, -(len(data) // -2), i)
        # plt.subplot(1, len(data), i)
        plt.hist(data[subset]["fraction"] * 100, density=True, bins=150, alpha=1.0)
        plt.title(subset)
        plt.ylabel("Normalized count")
        plt.xlabel("1-bp deletion frequency (%)")
        i += 1
    plt.suptitle("distributions")
    plt.tight_layout(pad=1.0)
    plt.show()


# Plot two subset distributions side-by-side
def compare_distributions(a, b, titles=["", ""]):
    plt.subplot(1, 2, 1)
    plt.hist(a["fraction"] * 100, density=True, bins=150, alpha=1.0)
    plt.title(titles[0])

    plt.subplot(1, 2, 2)
    plt.hist(b["fraction"] * 100, density=True, bins=150, alpha=1.0)
    plt.title(titles[1])

    plt.suptitle("distributions")
    plt.tight_layout(pad=1.0)
    plt.show()


# Plot of all subset distributions overlayed on top of each other
def distributions_combined(data):
    for subset in data:
        plt.hist(data[subset]["fraction"] * 100, bins=150, alpha=0.5)
    plt.title("distributions")
    plt.show()


# Plot of all subset distributions
def plot_distributions(data):
    for subset in data:
        plt.hist(data[subset]["fraction"] * 100, bins=150)
        plt.title(subset)
        plt.ylabel("Normalized count")
        plt.xlabel("1-bp deletion frequency (%)")
        plt.show()
