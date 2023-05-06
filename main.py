import random
import numpy as np
from numpy.random import randn
from pandas import DataFrame
from matplotlib import pyplot as plt
import pandas as pd
from time import process_time
from tqdm.auto import tqdm
import math

def data_generation(n_points=1000, subset_size=0.02):
    points_dict = {}

    for i in range(int(n_points / 2)):
        points_dict[(3 + randn(), 3 + randn())] = 1
        points_dict[(45 + randn(), 45 + randn())] = -1
        #points_dict[(3 + randn(), randn())] = 1
        #points_dict[(1.5 * randn(), 4 + randn())] = -1

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

    return 1 / (np.sqrt((x_1[0] - x_2[0]) ** 2 + (x_1[1] - x_2[1]) ** 2))

def similarity(x1, x2): # Euclidean
    dist = [(a - b)**2 for a, b in zip(x1, x2)]
    dist = math.sqrt(sum(dist))
    return dist

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

def calculate_accuracy(test_dict, pred_dict):
    correct = 0
    for key in test_dict.keys():
        if pred_dict[key] == test_dict[key]:
            correct += 1
    return correct / len(test_dict)

def calculate_loss(dict_nonlabeled, dict_labeled, dict_weights):
    # w:
    first_term_gradient = [(dict_weights[x_j, x_i] * (y_j - y_i) * (y_j - y_i)) for x_j, y_j in dict_nonlabeled.items() for x_i, y_i in dict_labeled.items()]
    # w_bar:
    second_term_gradient = [(dict_weights[x_j, x_i] * (y_j - y_i) * (y_j - y_i)) for x_j, y_j in dict_nonlabeled.items() for x_i, y_i in dict_nonlabeled.items() if x_i != x_j]
    return sum(first_term_gradient) + 0.5 * sum(second_term_gradient)

# Partial gradient (returns a scalar)
def gradient_yj(x_j, y_j, dict_unlabeled, dict_labeled, dict_weights):
    first_term_gradient  = [(dict_weights[x_j, x_i] * (y_j - y_i)) for x_i, y_i in dict_labeled.items()   if x_i != x_j]
    second_term_gradient = [(dict_weights[x_j, x_i] * (y_j - y_i)) for x_i, y_i in dict_unlabeled.items() if x_i != x_j]
    return 2 * (sum(first_term_gradient) + sum(second_term_gradient))

# Total/Full gradient (returns a vector)
def gradient_total(dict_unlabeled, dict_labeled, dict_weights):
    grad_list = []
    for x_j, y_j in dict_unlabeled.items():
        first_term_gradient = [(dict_weights[x_j, x_i] * (y_j - y_i)) for x_i, y_i in dict_labeled.items() if x_i != x_j]
        second_term_gradient = [(dict_weights[x_j, x_i] * (y_j - y_i)) for x_i, y_i in dict_unlabeled.items() if x_i != x_j]
        grad_list.append(2 * (sum(first_term_gradient) + sum(second_term_gradient)))
    return np.array(grad_list)

def rbcgd(dict_unlabeled, dict_labeled, dict_similarity_weights,
          iterations=1000, alpha=1e-2, grad_thres=1e-3):
    loss_list, cpu_time_list = [], []
    num_unlabelled = len(dict_unlabeled)
    U = np.zeros(num_unlabelled)
    print("Minimizing objective function:")
    for iteration in tqdm(range(iterations)):
        t_before = process_time()
        loss_list.append(calculate_loss(dict_unlabeled, dict_labeled, dict_similarity_weights))
        grad = gradient_total(dict_unlabeled, dict_labeled, dict_similarity_weights)
        random_pos = int(np.random.choice(num_unlabelled, 1))
        U[random_pos] = 1.0
        update_value = - alpha * U * grad
        array_unlabeled = np.array(list(dict_unlabeled.values()))
        array_unlabeled += update_value
        dict_unlabeled = dict(zip(dict_unlabeled.keys(), array_unlabeled))
        t_after = process_time()
        cpu_time_list.append(t_after - t_before)
        if np.linalg.norm(grad) < grad_thres:
            break
    return loss_list, cpu_time_list

def gradient_descent(dict_unlabeled, dict_labeled, dict_similarity_weights,
                     iterations=1000, alpha=1e-2, grad_thres=1e-3):
    stop_cond = False
    loss_list, cpu_time_list = [], []
    print("Minimizing objective function:")
    for iteration in range(iterations):
        print("Iteration:", iteration)
        t_before = process_time()
        loss_list.append(calculate_loss(dict_unlabeled, dict_labeled, dict_similarity_weights))
        for x_j, y_j in tqdm(dict_unlabeled.items()):
            grad = gradient_yj(x_j, y_j, dict_unlabeled, dict_labeled, dict_similarity_weights)
            if abs(grad) > grad_thres:
                dict_unlabeled[x_j] = y_j - alpha * grad
            else:
                stop_cond = True
                print(f'Current gradient value: {abs(grad)} < {grad_thres}.\nStopping at iteration {iteration}.')
                break
        t_after = process_time()
        cpu_time_list.append(t_after - t_before)
        if stop_cond:
            break
    return loss_list, cpu_time_list

def plot_loss(loss_list, algorithm, acc, alpha, grad_thres, cpu_time_list=[]):
    num_iterations = len(loss_list)
    fig, ax = plt.subplots()
    plt.grid(alpha=0.3)
    ax.set_title('{}\nAccuracy: {:.2f}%\nLearning Rate: {}\nGradient threshold: {}\nIterations: {}'.format(algorithm, acc * 100, alpha, grad_thres, num_iterations))
    ax.set_ylabel("Loss")
    if cpu_time_list:
        plt.plot(np.cumsum(cpu_time_list), loss_list, color='blue', marker='o', markerfacecolor='r')
        ax.set_xlabel("CPU time")
        filename = 'GD_acc{:.2f}_alpha{}_cpu_time{}.png'.format(acc, alpha, sum(loss_list))
    else:
        plt.plot(loss_list, color='blue', marker='o', markerfacecolor='r')
        ax.set_xlabel("Number of iterations")
        filename = 'GD_acc{:.2f}_alpha{}_num_iter{}.png'.format(acc, alpha, num_iterations)

    plt.savefig(filename)  # save the graph as an image with the parameters in the filename
    plt.show()

    # save the parameters and accuracy to a text file with the same filename as the image
    with open(filename.replace('.png', '.txt'), 'w') as f:
        f.write('Learning Rate: {}\n'.format(alpha))
        f.write('Iterations: {}\n'.format(num_iterations))
        f.write('Accuracy: {:.2f}%\n'.format(acc * 100))


if __name__ == '__main__':
    print("Start:")
    alpha = 1e-6
    grad_thres = 0.0005
    iterations = 10
    algorithms = ['Gradient Descent', 'RBCGD']
    algorithm = algorithms[1]
    dict_unlabeled, dict_labeled, dict_test, dict_similarity_weights = setup_points(n_points=1000, subset_size=0.01)
    loss_list, cpu_time_list = rbcgd(dict_unlabeled, dict_labeled, dict_similarity_weights,
                                     iterations=iterations,
                                     alpha=alpha,
                                     grad_thres=grad_thres)

    dict_unlabeled = {x_j: 1 if y_j >= 0 else -1 for x_j, y_j in dict_unlabeled.items()}
    all_points = {**dict_labeled, **dict_unlabeled}
    all_points_list = [[key[0], key[1]] for key in all_points.keys()]
    all_points_list = np.array(all_points_list)
    plotting(all_points_list, all_points.values())

    acc = calculate_accuracy(dict_test, dict_unlabeled)
    # Accuracy/Loss vs Iterations
    plot_loss(loss_list, algorithm, acc, alpha, grad_thres)
    # Accuracy/Loss vs CPU Time
    plot_loss(loss_list, algorithm, acc, alpha, grad_thres, cpu_time_list=cpu_time_list)
    print("Finished.")






