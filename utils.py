import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
from math import log

# Data Creation Functions
def data_creation(total_samples,unlabelled_ratio):

    # set the seed for reproducibility - in order to affect data point generation as well
    np.random.seed(10)

    # generate random data points with 2 features and 2 labels
    x, y = make_blobs(n_samples=total_samples, n_features=2, centers=2, cluster_std=1, random_state=10)
    y = 2 * y - 1

    # make %unlabelled_ratio of data points unlabeled
    num_unlabeled_samples = int(unlabelled_ratio * total_samples)

    # all_indices = labeled indices + unlabeled indices
    unlabeled_indices = np.random.choice(total_samples, size=num_unlabeled_samples, replace=False)
    labeled_indices = np.array(list(set(np.array(range(total_samples))) - set(unlabeled_indices)))

    weight_lu, weight_uu = create_similarity_matrices(x,labeled_indices,unlabeled_indices)

    return total_samples,unlabelled_ratio, x, y, unlabeled_indices,labeled_indices,weight_lu,weight_uu

def real_data(unlabelled_ratio = 0.9):

    # set the seed for reproducibility - in order to affect data point generation as well
    np.random.seed(10)

    data = np.genfromtxt('credit_card_defaulter.csv', delimiter=',', skip_header=1)

    total_samples = len(data)

    # make %unlabelled_ratio of data points unlabeled
    num_unlabeled_samples = int(unlabelled_ratio * total_samples)

    # all_indices = labeled indices + unlabeled indices
    unlabeled_indices = np.random.choice(total_samples, size=num_unlabeled_samples, replace=False)
    labeled_indices = np.array(list(set(np.array(range(total_samples))) - set(unlabeled_indices)))

    x = data[:, :2]
    y = data[:, -1]
    weight_lu, weight_uu = create_similarity_matrices(x,labeled_indices,unlabeled_indices)

    return total_samples, unlabelled_ratio, x, y, unlabeled_indices, labeled_indices, weight_lu, weight_uu

def create_similarity_matrices(x,labeled_indices,unlabeled_indices):
    eps = 1e-8  # not to get 0 in denominator
    weight_lu = 1 / (euclidean_distances(x[labeled_indices], x[unlabeled_indices]) + eps)
    weight_uu = 1 / (euclidean_distances(x[unlabeled_indices], x[unlabeled_indices]) + eps)

    return weight_lu, weight_uu

def plot_curves(y_list, x_list, x_label, y_label, title, legend, log_x=False, log_y=False):
    font = 16
    legend_size = 14
    label_size = 14

    _, ax = plt.subplots(figsize=(12, 9))
    for idx, y in enumerate(y_list):
        x = x_list[idx]
        if log_x:
            x = [log(i+1) for i in x]
        if log_y:
            y = [log(i + 1) for i in y]
        ax.plot(x, y,
                marker='o',
                linestyle='--',
                linewidth=1.3,
                markerfacecolor='white'
        )
        # If you want to change the span of the x-axis change the values here. Feel free to use fixed values as well.
        # if x_label == "CPU Time":
        #      ax.set_xlim(0, x[-1] * 0.5)
        # else:
        #      ax.set_xlim(0, len(x) * 0.5)
    #ax.set_xlim(1, 5)

    plt.legend(legend, prop={'size': legend_size})
    plt.title(title, fontsize=font)
    if log_x:
        x_label = "Log " + str(x_label)
    if log_y:
        y_label = "Log " + str(y_label)
    plt.xlabel(x_label, fontsize=font)
    plt.ylabel(y_label, fontsize=font)
    plt.tick_params(axis='both', labelsize=label_size)
    plt.grid()
    plt.show()

def plot_bar_per_model(result_df, metric="loss"):

    result_list = []
    for idx in result_df.index:
        if metric == "loss":
            result_list.append(result_df.loc[idx, "loss"][-1])
        elif metric == "accuracy":
            result_list.append(result_df.loc[idx, "accuracy"][-1])
        elif metric == "iterations":
            result_list.append(len(result_df.loc[idx, "loss"]))
        elif metric == "cpu_time":
            result_list.append(sum(result_df.loc[idx, "cpu_time"]))
        else:
            raise Exception("Wrong metric.")

    plt.figure(figsize=(12, 9))
    plt.bar(result_df["optim_alg"].tolist(), result_list)

    for i, v in enumerate(result_list):
        plt.text(i, v / 2, f'{v:.2f}', ha='center', va='center', rotation=90)

    # Set the x-axis ticks to show the index of each value in the list
    plt.xticks(range(len(result_list)), result_df["optim_alg"].tolist())

    # If you want to use a legend and 0, 1, ... on x-axis:
    # plt.xticks(range(len(result_list)), range(len(result_list)))
    # plt.legend(loc='upper right', labels=result_df["optim_alg"].tolist())

    # Set the axis labels and plot title
    plt.xlabel('Models')
    plt.ylabel(metric)
    plt.title(f'{metric} values for different models')
    plt.grid()
    plt.show()

def plot_bar_metrics(result_df):

    #result_df["Loss final"]     = 0.0
    result_df["Accuracy"]       = 0.0
    result_df["Iterations"]     = 0.0
    result_df["CPU total"]      = 0.0

    for idx in result_df.index:
        result_df.loc[idx, "Accuracy"]       = result_df.loc[idx, "accuracy"][-1]
        result_df.loc[idx, "Iterations"]     = len(result_df.loc[idx, "loss"])
        result_df.loc[idx, "CPU total"]      = sum(result_df.loc[idx, "cpu_time"])
        #result_df.loc[idx, "Loss final"]     = result_df.loc[idx, "loss"][-1]

    legend_list = result_df['optim_alg']
    result_df = result_df.drop(['optim_alg', 'loss', 'accuracy', 'cpu_time'], axis=1)

    # Scale to 0-1 range
    result_df_scaled = result_df.apply(lambda x: x/x.max(), axis=0)

    result_df_scaled = result_df_scaled.transpose()
    result_df        = result_df.transpose()

    ax = result_df_scaled.plot.bar(rot=0,
                                ylabel='Metrics results',
                                cmap='Paired',
                                figsize=(18, 16)
                                )

    for i, container in enumerate(ax.containers):
        raw_val = result_df.iloc[:, i]
        labels = [f"{r:.2f}" for r in raw_val]
        ax.bar_label(container, labels=labels)

    label_size = 15
    font_size = 17

    ax.set_title('Models performance comparison', fontsize=font_size)
    ax.set_ylabel('Metrics scores', fontsize=font_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.tick_params(axis='x', labelsize=label_size)
    ax.legend(loc='upper right', labels=legend_list)
    plt.grid()
    plt.show()
