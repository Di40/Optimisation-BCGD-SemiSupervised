import numpy as np
from numpy.random import randn
from matplotlib import pyplot as plt
from time import process_time
from tqdm.auto import tqdm
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
import datetime
import time


class Descent:
    """
    Class for implementing a descent algorithm for a semi-supervised learning task.

    Parameters:
    -----------
    total_samples : int, optional
        Total number of data points to generate (default is 1000).
    unlabelled_ratio : float, optional
        Ratio of data points to leave unlabeled (default is 0.9).
    learning_rate : float, optional
        Learning rate for the optimizer (default is 1e-5).
    threshold : float, optional
        Threshold for the optimizer to stop iterating (default is 1e-5).
    max_iterations : int, optional
        Maximum number of iterations for the optimizer to run (default is 100).

    Methods:
    --------
    create_data()
        Creates data points for the semi-supervised learning task.
    create_similarity_matrices()
        Creates similarity matrices for labeled and unlabeled data points.
    calculate_loss()
        Calculates the loss function for the current labels.
    calculate_accuracy()
        Calculates the accuracy of the current labels.
    calculate_gradient()
        Calculates the gradient for the current labels.
    optimize()
        Optimizes the labels using descent algorithm.
    plot_loss(save_plot=False)
        Plots the loss function over iterations.
    plot_accuracy(save_plot=False)
        Plots the accuracy over iterations.

    Attributes:
    -----------
    total_samples : int
        The total number of samples in the data set.
    unlabelled_ratio : float
        The proportion of the data set that is unlabelled (i.e., does not have a known output value).
    x : list
        The features of the data set.
    y : None
        The labels of the data set (if available).
    unlabeled_indices : list
        The indices of the unlabelled data points.
    labeled_indices : list
        The indices of the labelled data points.
    true_labels_of_unlabeled : list
        The true labels (if known) of the unlabelled data points.
    weight_lu : None
        The weight matrix for the labelled and unlabelled data.
    weight_uu : None
        The weight matrix for the unlabelled data only.
    learning_rate : float
        The step size used for gradient descent.
    threshold : float
        The convergence threshold for the algorithm.
    max_iterations : int
        The maximum number of iterations for the algorithm.
    name : str
        The name of the algorithm used for logging purposes.
    loss : list
        The loss function values at each iteration.
    cpu_time : list
        The CPU time used at each iteration.
    accuracy : list
        The accuracy of the model on the labelled data at each iteration.
    """

    def __init__(self, total_samples=1000, unlabelled_ratio=0.9, learning_rate=1e-5, threshold=1e-5,
                 max_iterations=100):

        # create data parameters
        self.total_samples = total_samples
        self.unlabelled_ratio = unlabelled_ratio
        self.x = []
        self.y = None
        self.unlabeled_indices = []
        self.labeled_indices = []
        self.true_labels_of_unlabeled = []

        # weight matrices
        self.weight_lu = None
        self.weight_uu = None

        # optimizor parameters
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.name = "Descent"

        # results parameters
        self.loss = []
        self.cpu_time = []
        self.accuracy = []

    def create_data(self):

        # set the seed for reproducibility - in order to affect data point generation as well
        np.random.seed(10)

        # generate random data points with 2 features and 2 labels
        self.x, self.y = make_blobs(n_samples=self.total_samples, n_features=2, centers=2, cluster_std=1,
                                    random_state=10)
        self.y = 2 * self.y - 1

        # plot the data points in a 2D scatter plot
        # plt.scatter(self.x[:, 0], self.x[:, 1])
        # plt.show()

        # make %90 of data points unlabeled
        num_unlabeled_samples = int(self.unlabelled_ratio * self.total_samples)

        # all_indices = labeled indices + unlabeled indices
        self.unlabeled_indices = np.random.choice(self.total_samples, size=num_unlabeled_samples, replace=False)
        self.labeled_indices = np.array(list(set(np.array(range(self.total_samples))) - set(self.unlabeled_indices)))

        # hold initially labeled then unlabeled points
        self.true_labels_of_unlabeled = np.copy(self.y[self.unlabeled_indices])

        self.plot_points(True)

        # assign initialization labels to unlabeled indices
        self.y = self.y.astype(float)
        self.y[self.unlabeled_indices] = np.random.uniform(-1.0, 1.0, size=num_unlabeled_samples)

    def create_similarity_matrices(self):
        eps = 1e-8  # not to get 0 in denominator
        self.weight_lu = 1 / (euclidean_distances(self.x[self.labeled_indices], self.x[self.unlabeled_indices]) + eps)
        self.weight_uu = 1 / (euclidean_distances(self.x[self.unlabeled_indices], self.x[self.unlabeled_indices]) + eps)

    def calculate_loss(self):
        Y_unlabeled = np.copy(self.y[self.unlabeled_indices]).astype("float32").reshape(
            (-1, 1))  # shape (len(unlabeled),1)
        Y_labeled = np.copy(self.y[self.labeled_indices]).astype("float32").reshape((-1, 1))  # shape (len(labeled),1)

        # Calculate first double sum
        y_diff = (Y_unlabeled - Y_labeled.T) ** 2  # shape (len(unlabeled),len(labeled))
        loss_lu = np.sum(y_diff * self.weight_lu.T)  # shape (len(unlabeled),len(labeled))

        # Calculate second double sum
        y_diff = (Y_unlabeled - Y_unlabeled.T) ** 2  # shape (len(unlabeled),len(unlabeled))
        loss_uu = np.sum(y_diff * self.weight_uu.T)  # shape (len(unlabeled),len(unlabeled))

        self.loss.append(loss_lu + loss_uu / 2)  # scalar

    def calculate_accuracy(self):
        rounded_y = np.where(self.y >= 0, 1, -1)
        num_correct = np.sum(np.round(rounded_y[self.unlabeled_indices]) == self.true_labels_of_unlabeled)
        # We want the values to be [-1, 1]. However, np.round(0.1) = 0.0, and we want that to be 1.0
        # num_correct = np.sum(np.round(self.y[self.unlabeled_indices]) == self.true_labels_of_unlabeled)
        self.accuracy.append(num_correct / len(self.true_labels_of_unlabeled))

    def calculate_gradient(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def optimize(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def plot_loss(self, save_plot):

        fig, ax = plt.subplots()
        plt.grid(alpha=0.3)
        ax.set_title(
            '{}\nAccuracy: {:.2f}%\nLearning Rate: {}\nGradient threshold: {}'
            .format(self.name,
                    self.accuracy[-1] * 100,
                    self.learning_rate,
                    self.threshold))
        ax.set_ylabel("Loss")
        ax.set_xlabel("Number of iterations")
        plt.plot(self.loss, color='blue', marker='o', markerfacecolor='r')

        if save_plot:
            now = datetime.datetime.now()
            time_str = now.strftime("%m.%d.2023-%H.%M")
            filename = 'LossPlot_{}_date {}, acc {:.2f}.png'.format(self.name, time_str, self.accuracy[-1] * 100)
            plt.savefig(filename)  # save the graph as an image with the parameters in the filename

        plt.show()

    def plot_accuracy(self, save_plot):

        fig, ax = plt.subplots()
        plt.grid(alpha=0.3)
        ax.set_title(
            '{}\nAccuracy: {:.2f}%\nLearning Rate: {}\nGradient threshold: {}'
            .format(self.name,
                    self.accuracy[-1] * 100,
                    self.learning_rate,
                    self.threshold))
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Number of iterations")
        plt.plot(self.accuracy, color='blue', marker='o', markerfacecolor='r')

        if save_plot:
            now = datetime.datetime.now()
            time_str = now.strftime("%m.%d.2023-%H.%M")
            filename = 'AccPlot_{}_date {}, acc {:.2f}.png'.format(self.name, time_str, self.accuracy[-1] * 100)
            plt.savefig(filename)  # save the graph as an image with the parameters in the filename

        plt.show()

    def plot_cpu_time(self, save_plot):

        fig, ax = plt.subplots()
        plt.grid(alpha=0.3)
        ax.set_title(
            '{}\nAccuracy: {:.2f}%\nLearning Rate: {}\nGradient threshold: {}'
            .format(self.name,
                    self.accuracy[-1] * 100,
                    self.learning_rate,
                    self.threshold))
        ax.set_ylabel("Loss")
        ax.set_xlabel("CPU Time")
        plt.plot(np.cumsum(self.cpu_time), self.loss, color='blue', marker='o', markerfacecolor='r')

        if save_plot:
            now = datetime.datetime.now()
            time_str = now.strftime("%m.%d.2023-%H.%M")
            filename = 'TimePlot_{}_date {}, acc {:.2f}.png'.format(self.name, time_str, self.accuracy[-1] * 100)
            plt.savefig(filename)  # save the graph as an image with the parameters in the filename

        plt.show()

    def save_output(self):

        now = datetime.datetime.now()
        time_str = now.strftime("%m.%d.2023-%H.%M")
        filename = '{}_date {}, acc {:.2f}.png'.format(self.name, time_str, self.accuracy[-1] * 100)

        number_labelled = self.total_samples - self.total_samples * self.unlabelled_ratio
        number_unlabelled = self.total_samples * self.unlabelled_ratio
        # save the parameters and accuracy to a text file with the same filename as the image
        with open(filename.replace('.png', '.txt'), 'w') as f:
            f.write('Learning Rate: {}\n'.format(self.learning_rate))
            f.write('Iterations: {}\n'.format(len(self.loss)))
            f.write('Loss: {}\n'.format(self.loss[-1]))
            f.write('Accuracy: {:.2f}%\n'.format(self.accuracy[-1] * 100))
            f.write('Number of Samples:{}\n'.format(self.total_samples))
            f.write('Number of Unlabelled-Labelled: {}-{}\n'.format(number_unlabelled, number_labelled))

    def plot_points(self, ul=False):  # TODO: rename flag and variables inside
        fig, ax = plt.subplots()
        if ul:  # show unlabelled points and labelled points together
            ax.scatter(self.x[:, 0], self.x[:, 1], color='black', marker=".", alpha=0.2)
            labeled_y = np.array(self.y)[self.labeled_indices]
            red = labeled_y == 1
            blue = labeled_y == -1
            ax.scatter(self.x[self.labeled_indices][red, 0], self.x[self.labeled_indices][red, 1], c="red",
                       marker=".")
            ax.scatter(self.x[self.labeled_indices][blue, 0], self.x[self.labeled_indices][blue, 1], c="blue",
                       marker=".")
            ax.set_title("Original points")
        else:  # show unlabelled points after optimization
            red = self.y >= 0
            blue = self.y < 0
            ax.scatter(self.x[red, 0], self.x[red, 1], c="red", marker=".")
            ax.scatter(self.x[blue, 0], self.x[blue, 1], c="blue", marker=".")
            ax.set_title("Predictions")
            # We can use this to plot the strength of the classification of each point:
            # ax.scatter(self.x[:, 0], self.x[:, 1], c=self.y, cmap='bwr', marker=".", alpha=0.2)

        plt.grid(alpha=0.3)
        plt.show()


class GradientDescent(Descent):
    def __init__(self, total_samples=1000, unlabelled_ratio=0.9, learning_rate=1e-5, threshold=1e-5,
                 max_iterations=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.name = "GradientDescent"

        self.total_samples = total_samples
        self.unlabelled_ratio = unlabelled_ratio

        self.gradient = []

    def calculate_gradient(self):
        # shape : (self.y[self.unlabeled_indices] -> (len,)
        # shape: (self.y[self.unlabeled_indices].reshape((-1,1)) -> (len,1)
        # This helps us to use broadcasting
        # shape : (self.y[self.unlabeled_indices].reshape((-1,1)) - self.y[self.labeled_indices] -> (len unlabelled, len labelled)

        weighted_diff = (self.y[self.unlabeled_indices].reshape((-1, 1)) - self.y[
            self.labeled_indices]) * self.weight_lu.T  # shape (len unlabelled, len labelled)
        grad_lu = np.sum(weighted_diff, axis=1)  # shape (len unlabelled, 1)  , sum all columns

        weighted_diff = (self.y[self.unlabeled_indices].reshape((-1, 1)) - self.y[
            self.unlabeled_indices]) * self.weight_uu.T  # shape (len unlabelled, len unlabelled)
        grad_uu = np.sum(weighted_diff, axis=1)  # shape (len unlabelled, 1)  , sum all columns

        self.gradient.append(grad_lu * 2 + grad_uu * 2)  # shape(len unlabelled,1)

    def optimize(self):

        ITERATION = 0

        while ITERATION < self.max_iterations:

            t_before = process_time()
            ITERATION += 1

            # Compute objective function for estimated y
            self.calculate_loss()
            self.calculate_accuracy()

            # Calculate gradient with respect to i
            self.calculate_gradient()

            # Update the estimated y
            self.y[self.unlabeled_indices] = self.y[self.unlabeled_indices] - self.learning_rate * self.gradient[-1]

            print("iteration: {} --- accuracy: {:.3f} ---- loss: {:.3f} --- next_stepsize: {}"
                  .format(ITERATION,
                          self.accuracy[-1],
                          self.loss[-1],
                          self.learning_rate))

            t_after = process_time()
            self.cpu_time.append(t_after - t_before)

            if abs(np.linalg.norm(np.array(self.gradient))) < self.threshold:
                break


class Randomized_BCGD(Descent):
    def __init__(self, total_samples=1000, unlabelled_ratio=0.9, learning_rate=1e-5, threshold=1e-5,
                 max_iterations=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.name = "R_BCGD"

        self.total_samples = total_samples
        self.unlabelled_ratio = unlabelled_ratio

        self.gradient = []

    def calculate_gradient(self, block):
        # shape grad_lu --> scalar
        grad_lu = np.sum((self.y[self.unlabeled_indices[block]] - self.y[
            self.labeled_indices])  # shape (scalar-vector number of labelled) = vector number of labelled
                         * self.weight_lu.T[
                             block])  # shape  vector num of labelled * vector num of labelled (for block)= vector num of labelled

        grad_uu = np.sum((self.y[self.unlabeled_indices[block]] - self.y[self.unlabeled_indices])
                         * self.weight_uu.T[block])  # shape vector num of unlabelled
        self.gradient.append(grad_lu * 2 + grad_uu * 2)

    def optimize(self):

        stop_condition = False
        ITERATION = 0

        while ITERATION < self.max_iterations:

            t_before = process_time()
            ITERATION += 1

            # Compute objective function for estimated y
            self.calculate_loss()
            self.calculate_accuracy()

            # Choosing random block
            rand_block = np.random.randint(len(self.unlabeled_indices))

            # Calculate gradient with respect to i
            self.calculate_gradient(rand_block)

            # Update the estimated y
            self.y[self.unlabeled_indices[rand_block]] = self.y[
                                                             self.unlabeled_indices[rand_block]] - self.learning_rate * \
                                                         self.gradient[-1]

            print("iteration: {} --- accuracy: {:.3f} ---- loss: {:.3f} --- next_stepsize: {}"
                  .format(ITERATION,
                          self.accuracy[-1],
                          self.loss[-1],
                          self.learning_rate))

            t_after = process_time()
            self.cpu_time.append(t_after - t_before)

            if abs(np.linalg.norm(np.array(self.gradient))) < self.threshold:  # TODO: Check whether this makes sense
                break


class GS_BCGD(Descent):
    def __init__(self, total_samples=1000, unlabelled_ratio=0.9, learning_rate=1e-5, threshold=1e-5,
                 max_iterations=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.name = "GS_BCGD"

        self.total_samples = total_samples
        self.unlabelled_ratio = unlabelled_ratio

        self.gradient = []

    def get_largest_gradient_index(self):

        weighted_diff = (self.y[self.unlabeled_indices].reshape((-1, 1)) - self.y[
            self.labeled_indices]) * self.weight_lu.T  # shape (len unlabelled, len labelled)
        grad_lu = np.sum(weighted_diff, axis=1)  # shape (len unlabelled, 1)  , sum all columns

        weighted_diff = (self.y[self.unlabeled_indices].reshape((-1, 1)) - self.y[
            self.unlabeled_indices]) * self.weight_uu.T  # shape (len unlabelled, len unlabelled)
        grad_uu = np.sum(weighted_diff, axis=1)  # shape (len unlabelled, 1)  , sum all columns

        full_grad = grad_lu * 2 + grad_uu  # shape(len unlabelled,1)
        max_grad_index = np.argmax(np.abs(full_grad))
        return full_grad[max_grad_index], max_grad_index

    def calculate_gradient(self, block):
        # shape grad_lu --> scalar
        grad_lu = np.sum((self.y[self.unlabeled_indices[block]] - self.y[
            self.labeled_indices])  # shape (scalar-vector number of labelled) = vector number of labelled
                         * self.weight_lu.T[
                             block])  # shape  vector num of labelled * vector num of labelled (for block)= vector num of labelled

        grad_uu = np.sum((self.y[self.unlabeled_indices[block]] - self.y[self.unlabeled_indices])
                         * self.weight_uu.T[block])  # shape vector num of unlabelled
        self.gradient.append(grad_lu * 2 + grad_uu * 2)

    def optimize(self):

        stop_condition = False
        ITERATION = 0

        while ITERATION < self.max_iterations:

            t_before = process_time()
            ITERATION += 1

            # Compute objective function for estimated y
            self.calculate_loss()
            self.calculate_accuracy()

            # Choosing max gradient block
            grad, max_gradient_index = self.get_largest_gradient_index()
            self.gradient.append(grad)

            # Update the estimated y
            self.y[self.unlabeled_indices[max_gradient_index]] = self.y[self.unlabeled_indices[
                max_gradient_index]] - self.learning_rate * self.gradient[-1]

            print("iteration: {} --- accuracy: {:.3f} ---- loss: {:.3f} --- next_stepsize: {}"
                  .format(ITERATION,
                          self.accuracy[-1],
                          self.loss[-1],
                          self.learning_rate))

            t_after = process_time()
            self.cpu_time.append(t_after - t_before)

            if abs(np.linalg.norm(np.array(self.gradient))) < self.threshold:  # TODO: Check whether this makes sense
                break


if __name__ == '__main__':

    # TODO: step size choice with Hessian
    # TODO: step size choice with Armijo

    #Save the current time
    # print("RBCGD Start")
    # start_time = time.time()
    # rbcgd = GS_BCGD(total_samples=1000,unlabelled_ratio=0.9,
    #                      learning_rate=1e-3,threshold=0.0001,max_iterations=5000)
    # rbcgd.create_data()
    # rbcgd.create_similarity_matrices()
    # rbcgd.optimize()
    # rbcgd.plot_points()
    # rbcgd.plot_loss(save_plot=True)
    # rbcgd.plot_accuracy(save_plot=True)
    # rbcgd.plot_cpu_time(save_plot=True)
    # rbcgd.save_output()
    #
    # elapsed_time = time.time() - start_time
    #
    # print(f"Time Spend:{elapsed_time}")
    #
    # print(f"*"*100)

    # print("Gradient Descent Start")
    # start_time = time.time()
    # gd = GradientDescent(total_samples=2000,unlabelled_ratio=0.9,
    #                      learning_rate=1e-5,threshold=0.0001,max_iterations=500)
    # gd.create_data()
    # gd.create_similarity_matrices()
    # gd.optimize()
    # gd.plot_points()
    # gd.plot_loss(save_plot=False)
    # gd.plot_accuracy(save_plot=False)
    # gd.plot_cpu_time(save_plot=False)
    # gd.save_output()
    #
    # elapsed_time = time.time() - start_time
    # print(f"Time Spend:{elapsed_time}")

    gd = GradientDescent(total_samples=2000, unlabelled_ratio=0.9,
                         learning_rate=1e-5, threshold=0.0001, max_iterations=500)
    gd.create_data()
    gd.create_similarity_matrices()
    # Gradient at point x
    gd.calculate_gradient()
    grad = gd.gradient[-1]  # (1800,1)
    # Search direction
    p = -np.array(grad)
    # Initialize counter for iterations
    iteration = 1
    # Set c for Armijo condition
    c = 0.9
    # Set roh for backtracking algorithm
    roh = 0.95
    while iteration < 20 \
            and np.linalg.norm(grad) > 10 ** -4:
        t_before = process_time()
        # Set alpha to 1 every time we enter the while loop
        # and calculate the new alpha for each iteration
        alpha = 0.05
        # Armijo condition in while loop
        while True:
            loss = gd.dedi_loss()
            loss += c * alpha * grad * p

            gd_modified = copy.deepcopy(gd)
            gd_modified.y = np.array(gd.y)
            gd_modified.y[gd_modified.unlabeled_indices] += alpha * p
            modified_loss = gd_modified.dedi_loss()

            sum_loss = np.sum(loss)
            sum_modified_loss = np.sum(modified_loss)
            if sum_modified_loss > sum_loss:
                alpha = roh * alpha
            else:
                break

        gd.calculate_gradient()
        grad = gd.gradient[-1]
        p = -np.array(grad)
        print(f"{iteration}, stepsize: {alpha}")
        gd.y[gd.unlabeled_indices] = np.array(gd.y[gd.unlabeled_indices]) + alpha * p

        gd.calculate_loss()
        gd.calculate_accuracy()
        t_after = process_time()
        gd.cpu_time.append(t_after - t_before)

        # Update iteration
        iteration += 1

    gd.plot_loss(save_plot=True)




    print("end")