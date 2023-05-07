import numpy as np
from numpy.random import randn
from pandas import DataFrame
from matplotlib import pyplot as plt
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
    # print("Start:")
    # alpha = 1e-6
    # grad_thres = 0.0005
    # iterations = 10
    # algorithms = ['Gradient Descent', 'RBCGD']
    # algorithm = algorithms[1]
    # dict_unlabeled, dict_labeled, dict_test, dict_similarity_weights = setup_points(n_points=1000, subset_size=0.01)
    # loss_list, cpu_time_list = rbcgd(dict_unlabeled, dict_labeled, dict_similarity_weights,
    #                                  iterations=iterations,
    #                                  alpha=alpha,
    #                                  grad_thres=grad_thres)
    #
    # dict_unlabeled = {x_j: 1 if y_j >= 0 else -1 for x_j, y_j in dict_unlabeled.items()}
    # all_points = {**dict_labeled, **dict_unlabeled}
    # all_points_list = [[key[0], key[1]] for key in all_points.keys()]
    # all_points_list = np.array(all_points_list)
    # plotting(all_points_list, all_points.values())
    #
    # acc = calculate_accuracy(dict_test, dict_unlabeled)
    # # Accuracy/Loss vs Iterations
    # plot_loss(loss_list, algorithm, acc, alpha, grad_thres)
    # # Accuracy/Loss vs CPU Time
    # plot_loss(loss_list, algorithm, acc, alpha, grad_thres, cpu_time_list=cpu_time_list)
    # print("Finished.")

    from sklearn.datasets import make_blobs
    from sklearn.metrics import euclidean_distances
    import datetime
    import time


    class Descent:
        def __init__(self,total_samples=1000,unlabelled_ratio=0.9, learning_rate=1e-5, threshold=1e-5,max_iterations=100):

            # create data parameters
            self.total_samples = total_samples
            self.unlabelled_ratio = unlabelled_ratio
            self.x = []
            self.y = None
            self.unlabeled_indices = []
            self.labeled_indices = []
            self.true_labels_of_unlabeled =[]

            # weight matrices
            self.weight_lu = None
            self.weight_uu = None

            # optimizor parameters
            self.learning_rate = learning_rate
            self.threshold = threshold
            self.max_iterations= max_iterations
            self.name = "Descent"

            # results parameters
            self.loss = []
            self.cpu_time = []
            self.accuracy  = []
            self.iterations_made = len(self.loss)



        def create_data(self):

            # generate random data points with 2 features and 2 labels
            self.x, self.y = make_blobs(n_samples=self.total_samples,n_features=2, centers=2, cluster_std=1,random_state = 10)


            # plot the data points in a 2D scatter plot
            plt.scatter(self.x[:, 0], self.x[:, 1])
            #plt.show()

            # set the seed for reproducibility
            np.random.seed(10)

            # make %90 of data points unlabeled
            num_unlabeled_samples = int(self.unlabelled_ratio * self.total_samples)

            # all_indices = labeled indices + unlabeled indices
            self.unlabeled_indices = np.random.choice(self.total_samples, size=num_unlabeled_samples, replace=False)
            self.labeled_indices = np.array(list(set(np.array(range(self.total_samples))) - set(self.unlabeled_indices)))

            # hold initially labeled then unlabeled points
            self.true_labels_of_unlabeled = np.copy(self.y[self.unlabeled_indices])

            # assign initialization labels to unlabeled indices
            self.y = self.y.astype(float)
            self.y[self.unlabeled_indices] = np.random.uniform(-1.0,1.0, size=num_unlabeled_samples)


        def create_similarity_matrices(self):
            eps = 1e-8  # not to get 0 in denominator
            self.weight_lu = 1 / (euclidean_distances(self.x[self.labeled_indices], self.x[self.unlabeled_indices]) + eps)
            self.weight_uu = 1 / (euclidean_distances(self.x[self.unlabeled_indices], self.x[self.unlabeled_indices]) + eps)


        def calculate_loss(self):
            Y_unlabeled = np.copy(self.y[self.unlabeled_indices]).astype("float32").reshape((-1, 1))  # shape (len(unlabeled),1)
            Y_labeled = np.copy(self.y[self.labeled_indices]).astype("float32").reshape((-1, 1))  # shape (len(labeled),1)

            # Calculate first double sum
            y_diff = np.power((Y_unlabeled - Y_labeled.T),2)  # shape (len(unlabeled),len(labeled))
            loss_lu = np.sum(y_diff * self.weight_lu.T)   # shape (len(unlabeled),len(labeled))

            # Calculate second double sum
            y_diff = np.power(Y_unlabeled - Y_unlabeled.T,2)  # shape (len(unlabeled),len(unlabeled))
            loss_uu = np.sum(y_diff * self.weight_uu.T)   # shape (len(unlabeled),len(unlabeled))

            self.loss.append(loss_lu + loss_uu/2) #scalar


        def calculate_accuracy(self):
            num_correct = np.sum(np.round(self.y[self.unlabeled_indices]) == self.true_labels_of_unlabeled)
            self.accuracy.append(num_correct / len(self.true_labels_of_unlabeled))


        def calculate_gradient(self):
            raise NotImplementedError("Subclass must implement abstract method")

        def optimize(self):
            raise NotImplementedError("Subclass must implement abstract method")

        def plot_loss(self,save_plot):

            fig, ax = plt.subplots()
            plt.grid(alpha=0.3)
            ax.set_title(
                '{}\nAccuracy: {:.2f}%\nLearning Rate: {}\nGradient threshold: {}\nIterations: {}'
                .format(self.name,
              self.accuracy[-1] * 100,
              self.learning_rate,
              self.threshold,
              self.iterations_made))
            ax.set_ylabel("Loss")
            ax.set_xlabel("Number of iterations")
            plt.plot(self.loss, color='blue', marker='o', markerfacecolor='r')

            if save_plot:
                now = datetime.datetime.now()
                time_str = now.strftime("%m.%d.2023-%H.%M")
                filename = 'LossPlot_{}_date {}, acc {}.png'.format(self.name, time_str, self.accuracy[-1] * 100)
                plt.savefig(filename)  # save the graph as an image with the parameters in the filename

            plt.show()

        def plot_accuracy(self,save_plot):

            fig, ax = plt.subplots()
            plt.grid(alpha=0.3)
            ax.set_title(
                '{}\nAccuracy: {:.2f}%\nLearning Rate: {}\nGradient threshold: {}\nIterations: {}'
                .format(self.name,
                        self.accuracy[-1] * 100,
                        self.learning_rate,
                        self.threshold,
                        self.iterations_made))
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Number of iterations")
            plt.plot(self.accuracy, color='blue', marker='o', markerfacecolor='r')

            if save_plot:
                now = datetime.datetime.now()
                time_str = now.strftime("%m.%d.2023-%H.%M")
                filename = 'AccPlot_{}_date {}, acc {}.png'.format(self.name, time_str, self.accuracy[-1] * 100)
                plt.savefig(filename)  # save the graph as an image with the parameters in the filename

            plt.show()

        def plot_cpu_time(self,save_plot):

            fig, ax = plt.subplots()
            plt.grid(alpha=0.3)
            ax.set_title(
                '{}\nAccuracy: {:.2f}%\nLearning Rate: {}\nGradient threshold: {}\nIterations: {}'
                .format(self.name,
                        self.accuracy[-1] * 100,
                        self.learning_rate,
                        self.threshold,
                        self.iterations_made))
            ax.set_ylabel("Loss")
            ax.set_xlabel("CPU Time")
            plt.plot(np.cumsum(self.cpu_time),self.loss, color='blue', marker='o', markerfacecolor='r')

            if save_plot:
                now = datetime.datetime.now()
                time_str = now.strftime("%m.%d.2023-%H.%M")
                filename = 'TimePlot_{}_date {}, acc {}.png'.format(self.name, time_str, self.accuracy[-1] * 100)
                plt.savefig(filename)  # save the graph as an image with the parameters in the filename

            plt.show()


        def save_output(self):

            now = datetime.datetime.now()
            time_str = now.strftime("%m.%d.2023-%H.%M")
            filename = '{}_date {}, acc {}.png'.format(self.name, time_str, self.accuracy[-1] * 100)

            number_labelled = self.total_samples - self.total_samples * self.unlabelled_ratio
            number_unlabelled = self.total_samples * self.unlabelled_ratio
            # save the parameters and accuracy to a text file with the same filename as the image
            with open(filename.replace('.png', '.txt'), 'w') as f:
                f.write('Learning Rate: {}\n'.format(self.learning_rate))
                f.write('Iterations: {}\n'.format(self.iterations_made))
                f.write('Loss: {}\n'.format(self.loss[-1]))
                f.write('Accuracy: {:.2f}%\n'.format(self.accuracy[-1] * 100))
                f.write('Number of Samples:{}\n'.format(self.total_samples))
                f.write('Number of Unlabelled-Labelled: {}-{}\n'.format(number_unlabelled, number_labelled))

    class GradientDescent(Descent):
        def __init__(self,total_samples=1000,unlabelled_ratio=0.9, learning_rate=1e-5, threshold=1e-5,max_iterations=100):
            super().__init__()
            self.learning_rate = learning_rate
            self.threshold = threshold
            self.max_iterations= max_iterations
            self.name="GradientDescent"

            self.total_samples = total_samples
            self.unlabelled_ratio = unlabelled_ratio

            self.gradient=[]

        def calculate_gradient(self,i):
            # shape : (self.y[self.unlabeled_indices] -> (len,)
            # shape: (self.y[self.unlabeled_indices].reshape((-1,1)) -> (len,1)
            # This helps us to use broadcasting
            # shape : (self.y[self.unlabeled_indices].reshape((-1,1)) - self.y[self.labeled_indices] -> (len unlabelled, len labelled)

            weighted_diff = (self.y[self.unlabeled_indices].reshape((-1,1)) - self.y[self.labeled_indices])* self.weight_lu.T # shape (len unlabelled, len labelled)
            grad_lu = np.sum(weighted_diff,axis=1) # shape (len unlabelled, 1)  , sum all columns

            weighted_diff = (self.y[self.unlabeled_indices].reshape((-1,1)) - self.y[self.unlabeled_indices])* self.weight_uu.T # shape (len unlabelled, len unlabelled)
            grad_uu = np.sum(weighted_diff,axis=1) # shape (len unlabelled, 1)  , sum all columns


            self.gradient.append(grad_lu * 2 + grad_uu) # shape(len unlabelled,1)

        def optimize(self):


            stop_condition = False
            ITERATION = 0

            while ITERATION < self.max_iterations:

                t_before = process_time()
                ITERATION += 1
                stop_condition = False

                # Check the stop condition
                if stop_condition:
                    break

                # Compute objective function for estimated y
                self.calculate_loss()
                self.calculate_accuracy()

                for i in range(len(self.unlabeled_indices)):

                    # Calculate gradient with respect to i
                    self.calculate_gradient(i)

                    # Stopping condition
                    #if abs(self.gradient[-1]) < self.threshold:
                    #    stop_condition = True

                    # Update the estimated y
                    self.y[self.unlabeled_indices] = self.y[self.unlabeled_indices] - self.learning_rate * self.gradient[-1]

                print("iteration: {} --- accuracy: {:.3f} ---- loss: {:.3f} --- next_stepsize: {}"
                      .format(ITERATION,
                      self.accuracy[-1],
                      self.loss[-1],
                      self.learning_rate)
                )
                t_after = process_time()
                self.cpu_time.append(t_after - t_before)


    class Randomized_BCGD(Descent):
        def __init__(self,total_samples=1000,unlabelled_ratio=0.9, learning_rate=1e-5, threshold=1e-5,max_iterations=100):
            super().__init__()
            self.learning_rate = learning_rate
            self.threshold = threshold
            self.max_iterations= max_iterations
            self.name="BCGD"

            self.total_samples = total_samples
            self.unlabelled_ratio = unlabelled_ratio

            self.gradient=[]

        def calculate_gradient(self, block):
            # shape grad_lu --> scalar
            grad_lu = np.sum((self.y[self.unlabeled_indices[block]] - self.y[self.labeled_indices])  # shape (scalar-vector number of labelled) = vector number of labelled
                             * self.weight_lu.T[block])  # shape  vector num of labelled * vector num of labelled (for block)= vector num of labelled

            grad_uu = np.sum((self.y[self.unlabeled_indices[block]] - self.y[self.unlabeled_indices])
                             * self.weight_uu.T[block]) # shape vector num of unlabelled
            self.gradient.append(grad_lu * 2 + grad_uu)

        def optimize(self):

            stop_condition = False
            ITERATION = 0

            while ITERATION < self.max_iterations:

                t_before = process_time()
                ITERATION += 1
                stop_condition = False

                # Check the stop condition
                if stop_condition:
                    break

                # Compute objective function for estimated y
                self.calculate_loss()
                self.calculate_accuracy()

                for _ in range(len(self.unlabeled_indices)):

                    # Choosing random block
                    rand_block = np.random.randint(len(self.unlabeled_indices))

                    # Calculate gradient with respect to i
                    self.calculate_gradient(rand_block)

                    # Stopping condition
                    if abs(self.gradient[-1]) < self.threshold:
                        stop_condition = True

                    # Update the estimated y
                    self.y[self.unlabeled_indices[rand_block]] = self.y[self.unlabeled_indices[rand_block]] - self.learning_rate * self.gradient[-1]

                print("iteration: {} --- accuracy: {:.3f} ---- loss: {:.3f} --- next_stepsize: {}"
                      .format(ITERATION,
                      self.accuracy[-1],
                      self.loss[-1],
                      self.learning_rate)
                )
                t_after = process_time()
                self.cpu_time.append(t_after - t_before)




    # TODO: plot function to show unlabelled points graph
    # TODO: class Gauss Sauthwell BCGD
    # TODO: write docstrings for classes and functions
    # TODO: step size choice with different methods
    # TODO: threshold does not seem so logical, we should consider smarter way

    # Save the current time
    print("RBCGD Start")
    start_time = time.time()
    rbcgd = Randomized_BCGD(total_samples=2000,unlabelled_ratio=0.9,
                         learning_rate=1e-5,threshold=0.0001,max_iterations=50)
    rbcgd.create_data()
    rbcgd.create_similarity_matrices()
    rbcgd.optimize()
    rbcgd.plot_loss(save_plot=True)
    rbcgd.plot_accuracy(save_plot=True)
    rbcgd.plot_cpu_time(save_plot=True)
    rbcgd.save_output()

    elapsed_time = time.time() - start_time

    print(f"Time Spend:{elapsed_time}")

    print(f"*"*100)

    # print("Gradient Descent Start")
    # start_time = time.time()
    # gd = GradientDescent(total_samples=2000,unlabelled_ratio=0.9,
    #                      learning_rate=1e-5,threshold=0.0001,max_iterations=500)
    # gd.create_data()
    # gd.create_similarity_matrices()
    # gd.optimize()
    # gd.plot_loss(save_plot=True)
    # gd.plot_accuracy(save_plot=True)
    # gd.save_output()
    #
    # elapsed_time = time.time() - start_time
    # print(f"Time Spend:{elapsed_time}")




    print("end")