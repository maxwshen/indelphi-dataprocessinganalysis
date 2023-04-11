import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


# Used for testing with dummy data
def dummy_scatterplot():
    # Dummy dataset size
    number_of_gRNAs = 1000
    number_of_samples_per_gRNA = 50

    # X-axis layout
    x_ticks = [20, 45, 70, 95]
    base_ticks = [-6, -2, 2, 6]

    # Generate dummy data
    gRNAs = []
    nucleotide_4_per_gRNA = []
    insertion_per_gRNA = []
    color = ["r", "b", "gold", "g"]
    for i in range(number_of_gRNAs):
        gRNAs.append(i)
        min4 = []
        insertion = []
        for j in range(number_of_samples_per_gRNA):
            nucleotide = 4 * random.random()
            insert = 4 * random.random()
            # Randomly pick the -4 nucleotide
            if nucleotide < 1.0:
                min4.append("A")
            elif nucleotide < 2.0:
                min4.append("T")
            elif nucleotide < 3.0:
                min4.append("C")
            else:
                min4.append("G")

            # Artificial bias for -4 = A
            if min4[j] == "A":
                if random.random() < 0.5:
                    insertion.append("A")
                elif insert < 1.33:
                    insertion.append("T")
                elif insert < 2.0:
                    insertion.append("C")
                else:
                    insertion.append("G")

            else:
                # Randomly pick the inserted nucleotide for -4 = T/C/G
                if insert < 1.0:
                    insertion.append("A")
                elif insert < 2.0:
                    insertion.append("T")
                elif insert < 3.0:
                    insertion.append("C")
                else:
                    insertion.append("G")

        nucleotide_4_per_gRNA.append(min4)
        insertion_per_gRNA.append(insertion)

    # We now have list of gRNAs
    # For each gRNA, we have lists of samples with a specific -4 and insertion nucleotide

    # For each gRNA, we now have to determine the percentage of insertion nucleotides per -4 nucleotide

    # [[x_positions], [frequencies], [colors]]
    datapoints = [[], [], []]

    # For each gRNA, compute frequencies of insertions per -4 nucleotide
    for i in range(number_of_gRNAs):
        samples_min4 = nucleotide_4_per_gRNA[i]
        samples_ins = insertion_per_gRNA
        min4_sorted = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

        # For each sample, check the -4 nucleotide and update corresponding frequency array
        for j in range(number_of_samples_per_gRNA):
            if nucleotide_4_per_gRNA[i][j] == "A":
                index = 0
            elif nucleotide_4_per_gRNA[i][j] == "T":
                index = 1
            elif nucleotide_4_per_gRNA[i][j] == "G":
                index = 2
            else:
                index = 3

            if insertion_per_gRNA[i][j] == "A":
                min4_sorted[index][0] += 1.0
            elif insertion_per_gRNA[i][j] == "T":
                min4_sorted[index][1] += 1.0
            elif insertion_per_gRNA[i][j] == "C":
                min4_sorted[index][2] += 1.0
            elif insertion_per_gRNA[i][j] == "G":
                min4_sorted[index][3] += 1.0

        # Convert counts to frequencies
        for k in range(len(min4_sorted)):
            min4_sorted[k] = np.divide(min4_sorted[k], sum(min4_sorted[k])) * 100

            # Create datapoints for this gRNA
            for l in range(len(min4_sorted[k])):
                datapoints[0].append(x_ticks[k] + (base_ticks[l]) + (random.random() - 0.5))
                datapoints[1].append(min4_sorted[k][l])
                datapoints[2].append(color[l])

    # Plotting the dataset
    # Bin data for boxplot
    bins = []
    for i in range(len(min4_sorted)):
        bins.append(bin_data_per_min4(datapoints,
                                      x_ticks[i] + min(base_ticks) - 1,
                                      x_ticks[i] - min(base_ticks) + 1,
                                      color))

    # Plot boxplots
    for i in range(len(bins)):
        for j in range(len(bins)):
            plt.boxplot(bins[i][j], positions=[x_ticks[i] + base_ticks[j]], widths=3, showfliers=False)

    # Plot datapoints
    plt.scatter(datapoints[0], datapoints[1], c=datapoints[2], s=5.0, alpha=0.1)
    plt.xticks(x_ticks, ["A", "T", "C", "G"])
    plt.xlim(left=10)
    legend_markers = [Line2D([0], [0], lw=4, marker="o", color="w", markerfacecolor="r", markersize=10),
                      Line2D([0], [0], lw=4, marker="o", color="w", markerfacecolor="b", markersize=10),
                      Line2D([0], [0], lw=4, marker="o", color="w", markerfacecolor="gold", markersize=10),
                      Line2D([0], [0], lw=4, marker="o", color="w", markerfacecolor="g", markersize=10)]
    plt.legend(legend_markers, ["A", "T", "C", "G"], loc=1)
    plt.xlabel("Nucleotide at position -4")
    plt.ylabel("Frequency of inserted base among 1-bp ins., dummy data (%)")
    plt.title("Frequency of inserted base for different -4 position bases")
    plt.show()


# Helper function for dummy plot
def bin_data_per_min4(datapoints, x_min, x_max, colors):
    current_bin = [[], [], [], []]
    for i in range(len(datapoints[0])):
        if (datapoints[0][i] > x_min) & (datapoints[0][i] < x_max):
            if datapoints[2][i] == colors[0]:
                current_bin[0].append(datapoints[1][i])
            if datapoints[2][i] == colors[1]:
                current_bin[1].append(datapoints[1][i])
            if datapoints[2][i] == colors[2]:
                current_bin[2].append(datapoints[1][i])
            if datapoints[2][i] == colors[3]:
                current_bin[3].append(datapoints[1][i])

    return current_bin


# Helper function for confidence intervals
def plot_confidence_interval(x, values, z=1.96, color='k', notch_size=2, line_width=1.5):
    mean = np.mean(values)
    stdev = np.std(values)
    confidence_interval = z * stdev / np.sqrt(len(values))

    left = x - notch_size / 2
    top = mean - confidence_interval
    right = x + notch_size / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color, linewidth=line_width)
    plt.plot([left, right], [top, top], color=color, linewidth=line_width)
    plt.plot([left, right], [bottom, bottom], color=color, linewidth=line_width)
    plt.plot(x, mean, "_", markersize=15, color='k')

    return mean, confidence_interval


# Finalized function for recreating Figure 2a using the same data as the paper
def recreate_fig_2a(min4_base, a_frac, t_frac, c_frac, g_frac):
    # Constants for visualization
    x_ticks = [20, 45, 70, 95]
    base_ticks = [-6, -2, 2, 6]
    colors = ["tab:red", "tab:blue", "gold", "tab:green"]
    bases = ["A", "T", "C", "G"]

    # Merge insertion fractions per gRNA
    frequencies = []
    for i in range(len(a_frac)):
        frequencies.append([a_frac[i] * 100, t_frac[i] * 100, c_frac[i] * 100, g_frac[i] * 100])

    # Create a datapoint for each frequency per gRNA with following attributes:
    # [[x_position], [frequency], [inserted_nucleotide], [color]]
    data = [[], [], [], []]
    for i in range(len(min4_base)):
        for j in range(len(frequencies[0])):
            data[0].append(x_ticks[bases.index(min4_base[i])] + (base_ticks[j]) + (random.random() - 0.5))
            data[1].append(frequencies[i][j])
            data[2].append(min4_base[i])
            data[3].append(colors[j])

    # Bin data by x-position for statistics
    bins = []
    for i in range(len(bases)**2):
        bins.append([])
    for i in range(len(data[0])):
        binned = False
        for j in range(len(x_ticks)):
            if not binned:
                for k in range(len(base_ticks)):
                    if not binned:
                        if data[0][i] < x_ticks[j] + base_ticks[k] + 1:
                            bins[4 * j + k].append(data[1][i])
                            binned = True

    # Plot confidence intervals/boxplots
    for i in range(len(x_ticks)):
        for j in range(len(base_ticks)):
            plot_confidence_interval(x_ticks[i] + base_ticks[j], bins[i * 4 + j])
            # plt.boxplot(bins[i * 4 + j], positions=[x_ticks[i] + base_ticks[j]], widths=3, showfliers=False)

    # Plot datapoints
    plt.scatter(data[0], data[1], c=data[3], s=5.0, alpha=0.2)
    plt.xticks(x_ticks, bases)
    plt.xlim(left=10)
    legend_markers = [Line2D([0], [0], lw=4, marker="o", color="w", markerfacecolor=colors[0], markersize=10),
                      Line2D([0], [0], lw=4, marker="o", color="w", markerfacecolor=colors[1], markersize=10),
                      Line2D([0], [0], lw=4, marker="o", color="w", markerfacecolor=colors[2], markersize=10),
                      Line2D([0], [0], lw=4, marker="o", color="w", markerfacecolor=colors[3], markersize=10)]
    plt.legend(legend_markers, bases, loc=1)
    plt.xlabel("Nucleotide at position -4")
    plt.ylabel("Frequency of inserted base among 1-bp ins., lib-A mESCs (%)")
    plt.title("Frequency of inserted base for different -4 position bases")
    plt.show()


if __name__ == "__main__":
    # dummy_scatterplot()

    data = pd.read_excel("41586_2018_686_MOESM4_ESM.xlsx", sheet_name=0)
    recreate_fig_2a(data["Base"], data["A frac"], data["T frac"], data["C frac"], data["G frac"])
