import random
import numpy as np
from numpy.random import randn
from pandas import DataFrame
from matplotlib import pyplot as plt
import pandas as pd
from time import process_time
from tqdm.auto import tqdm

def data_generation(n_points=1000, subset_size=0.02):
    points_dict = {}

    for i in range(int(n_points / 2)):
        # points_dict[(3 + randn(), 3 + randn())] = 1
        # points_dict[(4.5 + randn(), 4.5 + randn())] = -1
        points_dict[(3 + randn(), randn())] = 1
        points_dict[(1.5 * randn(), 4 + randn())] = -1

    points_list = [key for key in points_dict.keys()]
    labels_list = [value for value in points_dict.values()]

    len_labeled_dataset = int(subset_size * len(points_list))

    labeled_points = points_list[:len_labeled_dataset]
    labels = labels_list[:len_labeled_dataset]
    non_labeled_points = points_list[len_labeled_dataset:]
    test_labels = labels_list[len_labeled_dataset:]

    print(f"Generated data with {n_points} points and {100 * subset_size}% of label points")

    plotting(np.array(points_list), np.array(labels_list))
    plotting(np.array(labeled_points), np.array(labels), np.array(non_labeled_points), nl=True)

    return labeled_points, non_labeled_points, labels, test_labels

def setup_points(n_points=1000, subset_size=0.02):
    labeled_points, non_labeled_points, labels, labels_check = data_generation(n_points, subset_size) # lists
    dummy_labels = np.zeros(len(non_labeled_points))
    dict_labeled = dict(zip(labeled_points, labels))
    dict_nonlabeled = dict(zip(non_labeled_points, dummy_labels))
    dict_test = dict(zip(non_labeled_points, labels_check))
    print("Computing similarity matrix...")
    dict_weights_similarity = calculate_similarity_weights({**dict_labeled, **dict_nonlabeled})
    print("Similarity matrix constructed.")

    return dict_nonlabeled, dict_labeled, dict_test, dict_weights_similarity


def similarity(x_1, x_2):
    return 1 / (np.sqrt((x_1[0] - x_2[0]) ** 2 + (x_1[1] - x_2[1]) ** 2))

def calculate_similarity_weights(dict_points):
    dict_similarity_weights = {}
    for point1 in tqdm(dict_points.keys()):
        for point2 in dict_points.keys():
            if point1 != point2:
                dict_similarity_weights[(point1, point2)] = similarity(point1, point2)
    return dict_similarity_weights


def plotting(points, labels, non_labeled_points={}, nl=False):
    if nl:
        figure, axis = plt.subplots()
        plt.scatter(non_labeled_points[:, 0], non_labeled_points[:, 1], color='black', marker=".", alpha=0.2)
    else:
        figure, axis = plt.subplots()

    df = DataFrame(dict(x=points[:, 0], y=points[:, 1], label=labels))
    colors = {-1: 'red', 1: 'blue'}
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=axis, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.show()


def calculate_loss(dict_nonlabeled, dict_labeled, dict_weights):
    # w:
    first_term_gradient = [(dict_weights[x_j, x_i] * (y_j - y_i) * (y_j - y_i)) for x_j, y_j in dict_nonlabeled.items() for x_i, y_i in dict_labeled.items()]
    # w_bar:
    second_term_gradient = [(dict_weights[x_j, x_i] * (y_j - y_i) * (y_j - y_i)) for x_j, y_j in dict_nonlabeled.items() for x_i, y_i in dict_nonlabeled.items() if x_i != x_j]
    return sum(first_term_gradient) + 0.5 * sum(second_term_gradient)

def gradient_descent(dict_unlabeled, dict_labeled, dict_test, dict_similarity_weights, alpha= 1e-4, grad_thres=0.005, to_plot=True, rbcgd=False):
    stop_cond = False
    loss_list, cpu_time = [], []
    max_iter = 1000

    for iteration in tqdm(range(max_iter)):
        t_before = process_time()
        loss_list.append(calculate_loss(dict_unlabeled, dict_labeled, dict_similarity_weights))
        for x_j, y_j in dict_unlabeled.items():
            grad = gradient_yj(x_j, y_j, dict_unlabeled, dict_labeled, dict_similarity_weights, rbcgd)
            if abs(grad) > grad_thres:
                dict_unlabeled[x_j] = y_j - alpha * grad
            else:
                stop_cond = True
                print(f'Current gradient value: {abs(grad)} < {grad_thres}.\nStopping at iteration {iteration}.')
                break
        t_after = process_time()
        cpu_time.append(t_after - t_before)
        if stop_cond:
            break

    for x_j, y_j in dict_unlabeled.items():
        if y_j >= 0:
            dict_unlabeled[x_j] = 1
        else:
            dict_unlabeled[x_j] = -1

    print(f"Unlabeled points have been classified with an accuracy of {calculate_accuracy(dict_test, dict_unlabeled)}.")

    #### Plotting part ##############
    if to_plot == True:
        all_points = {**dict_labeled, **dict_unlabeled}
        all_points_list = [[key[0], key[1]] for key in all_points.keys()]
        all_points_list = np.array(all_points_list)
        plotting(all_points_list, all_points.values())

    return loss_list, cpu_time


def gradient_yj(x_j, y_j, dict_unlabeled, dict_labeled, dict_weights, rbcgd=False):
    if rbcgd:
        feature = int(np.random.choice(2, 1))
        first_term_gradient = [(1 / abs(x_j[feature] - x_i[feature]) * (y_j - y_i)) for x_i, y_i in dict_labeled.items() if x_i[feature] != x_j[feature]]
        second_term_gradient = [(1 / abs(x_j[feature] - x_i[feature]) * (y_j - y_i)) for x_i, y_i in dict_unlabeled.items() if x_i[feature] != x_j[feature]]
    else:
        first_term_gradient = [(dict_weights[x_j, x_i] * (y_j - y_i)) for x_i, y_i in dict_labeled.items() if x_i != x_j]
        second_term_gradient = [(dict_weights[x_j, x_i] * (y_j - y_i)) for x_i, y_i in dict_unlabeled.items() if x_i != x_j]
    return 2 * (sum(first_term_gradient) + sum(second_term_gradient))

def calculate_accuracy(test_dict, pred_dict):
    correct = 0
    for key in test_dict.keys():
        if pred_dict[key] == test_dict[key]:
            correct += 1
    return correct / len(test_dict)


if __name__ == '__main__':
    print("Start:")
    dict_unlabeled, dict_labeled, dict_test, dict_similarity_weights = setup_points(n_points=6000, subset_size=0.01)

    func_values_gdo, cpu_time_gdo = gradient_descent(dict_unlabeled, dict_labeled, dict_test, dict_similarity_weights)
                                                        # alpha = 1e-3, grad_thres = 0.005, to_plot = True, rbcgd=False)

    # Replace the following code with a function
    fig, ax = plt.subplots()
    plt.grid(alpha = 0.2)
    plt.plot(func_values_gdo, color='blue', marker = 'o', markerfacecolor = 'r')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Number of iterations")
    plt.show()
    print("Finished.")

