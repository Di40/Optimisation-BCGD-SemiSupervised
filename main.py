import numpy as np
from numpy.random import randn
from matplotlib import pyplot as plt
from time import process_time
from tqdm.auto import tqdm
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
import datetime
import time
import copy


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

    def __init__(self,learning_rate=1e-5,max_iterations=100):

        # create data parameters
        self.total_samples = None
        self.unlabelled_ratio = None
        self.x = []
        self.y = []
        self.unlabeled_indices = []
        self.labeled_indices = []
        self.true_labels_of_unlabeled = []

        # weight matrices
        self.weight_lu = None
        self.weight_uu = None

        # optimizor parameters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.name = "Descent"

        # results parameters
        self.loss = []
        self.cpu_time = []
        self.accuracy = []

    def load_data(self,total_samples,unlabelled_ratio,x,y,unlabeled_indices,labeled_indices,weight_lu,weight_uu):

        self.total_samples = total_samples
        self.unlabelled_ratio = unlabelled_ratio
        self.x = x
        self.y = y
        self.unlabeled_indices = unlabeled_indices
        self.labeled_indices = labeled_indices
        self.weight_lu = weight_lu
        self.weight_uu = weight_uu

        # hold initially labeled then unlabeled points
        self.true_labels_of_unlabeled  = np.copy(self.y[self.unlabeled_indices])

        self.plot_points(True)

        # assign initialization labels to unlabeled indices
        self.y = self.y.astype(float)
        self.y[self.unlabeled_indices] = np.random.uniform(-1.0, 1.0, size=len(self.unlabeled_indices))


    # def create_data(self):
    #
    #     # set the seed for reproducibility - in order to affect data point generation as well
    #     np.random.seed(10)
    #
    #     # generate random data points with 2 features and 2 labels
    #     self.x, self.y = make_blobs(n_samples=self.total_samples, n_features=2, centers=2, cluster_std=1,
    #                                 random_state=10)
    #     self.y = 2 * self.y - 1
    #
    #     # make %90 of data points unlabeled
    #     num_unlabeled_samples = int(self.unlabelled_ratio * self.total_samples)
    #
    #     # all_indices = labeled indices + unlabeled indices
    #     self.unlabeled_indices = np.random.choice(self.total_samples, size=num_unlabeled_samples, replace=False)
    #     self.labeled_indices = np.array(list(set(np.array(range(self.total_samples))) - set(self.unlabeled_indices)))
    #
    #     # hold initially labeled then unlabeled points
    #     self.true_labels_of_unlabeled = np.copy(self.y[self.unlabeled_indices])
    #
    #     self.plot_points(True)
    #
    #     # assign initialization labels to unlabeled indices
    #     self.y = self.y.astype(float)
    #     self.y[self.unlabeled_indices] = np.random.uniform(-1.0, 1.0, size=num_unlabeled_samples)
    # def create_similarity_matrices(self):
    #     eps = 1e-8  # not to get 0 in denominator
    #     self.weight_lu = 1 / (euclidean_distances(self.x[self.labeled_indices], self.x[self.unlabeled_indices]) + eps)
    #     self.weight_uu = 1 / (euclidean_distances(self.x[self.unlabeled_indices], self.x[self.unlabeled_indices]) + eps)


    def calculate_loss(self,y_labelled, y_unlabelled):

        Y_labeled = np.copy(y_labelled).astype("float32").reshape((-1, 1))  # shape (len(labeled),1)
        Y_unlabeled = np.copy(y_unlabelled).astype("float32").reshape((-1, 1))  # shape (len(unlabeled),1)

        # Calculate first double sum
        y_diff = (Y_unlabeled - Y_labeled.T) ** 2  # shape (len(unlabeled),len(labeled))
        loss_lu = np.sum(y_diff * self.weight_lu.T)  # shape (len(unlabeled),len(labeled))

        # Calculate second double sum
        y_diff = (Y_unlabeled - Y_unlabeled.T) ** 2  # shape (len(unlabeled),len(unlabeled))
        loss_uu = np.sum(y_diff * self.weight_uu.T)  # shape (len(unlabeled),len(unlabeled))

        return (loss_lu + loss_uu / 2)  # scalar

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
            '{}\nAccuracy: {:.2f}%\nLearning Rate: {}\nGradient'
            .format(self.name,
                    self.accuracy[-1] * 100,
                    self.learning_rate))
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
            '{}\nAccuracy: {:.2f}%\nLearning Rate: {}\n'
            .format(self.name,
                    self.accuracy[-1] * 100,
                    self.learning_rate))
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
            '{}\nAccuracy: {:.2f}%\nLearning Rate: {}\n'
            .format(self.name,
                    self.accuracy[-1] * 100,
                    self.learning_rate))
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

    def plot_points(self, ul=False):
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

    def early_stopping(self):
        delta_new = self.loss[-1] - self.loss[-2]
        delta_old = self.loss[-2] - self.loss[-3]
        if delta_new/ delta_old < 0.01:
            return True

class Gradient_Descent(Descent):
    def __init__(self, learning_rate=1e-5, threshold=1e-5,
                 max_iterations=100,learning_rate_strategy='constant'):
        super().__init__()
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.name = "GradientDescent"

        self.gradient = []
        self.learning_rate_strategy = learning_rate_strategy


    # Learning rate arranger algorithms
    def _armijo_rule(self, alpha = 0.05, delta = 0.95, gamma = 0.49):
        """
        Args:
        - alpha: float, the initial learning rate
        - delta: float, constant in (0,1) representing the proportion by which we decrease the learning rate
        - gamma: float, constant in (0,1/2) representing the decrease rate of the learning rate
        - grad: numpy array of shape (n_features,), the gradients of the loss function with respect to the weights
        """
        # Compute the initial loss
        current_loss = self.loss[-1]

        # Compute the squared norm of the gradient
        grad = self.gradient[-1]  # (len unlabelled,1)
        dk = -grad

        # Iterate up to max_iterion
        while True:
            # Check the Armijo condition, i.e., if the expected loss is smaller than the current loss plus some threshold
            expected_loss = self.calculate_loss(self.y[self.labeled_indices],self.y[self.unlabeled_indices] + alpha * dk)
            #print(f"expected loss:{expected_loss}, current_loss+change:{current_loss + gamma * alpha * np.dot(grad.T, dk)} ")
            if  expected_loss <= current_loss + gamma * alpha * np.dot(grad.T, dk) :
                # If the condition is met, return the chosen learning rate
                return alpha
            else:
                # If the condition is not met, decrease the learning rate by a constant factor
                alpha *= delta

    def _Lipschitz_constant(self):
        eig_vals, _ = np.linalg.eig(self._hessian_matrix())
        return max(eig_vals)

    def _hessian_matrix(self):
        h_mat = np.copy(-self.weight_uu)
        for i in range(len(self.unlabeled_indices)):
            h_mat[i][i] += 2 * (np.sum(self.weight_lu[:, i]) + np.sum(self.weight_uu[:, i])) - self.weight_uu[i, i]
        return h_mat

    def _learning_rate(self):

        if self.learning_rate_strategy == 'constant' or self.learning_rate_strategy == 0 :
            return self.learning_rate
        elif self.learning_rate_strategy == 'armijo' or self.learning_rate_strategy == 1:
            lr = self._armijo_rule()
            return lr
        elif self.learning_rate_strategy == 'lipschitz' or self.learning_rate_strategy == 2:
            L = self._Lipschitz_constant()
            return 1 / L



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
            self.loss.append(self.calculate_loss(self.y[self.labeled_indices],self.y[self.unlabeled_indices]))
            self.calculate_accuracy()

            # Calculate gradient with respect to i
            self.calculate_gradient()

            # Choose learning rate
            self.learning_rate = self._learning_rate()

            # Update the estimated y
            self.y[self.unlabeled_indices] = self.y[self.unlabeled_indices] - self.learning_rate * self.gradient[-1]

            print("{} iteration: {} --- accuracy: {:.3f} ---- loss: {:.3f} --- next_stepsize: {}"
                  .format(np.linalg.norm(np.array(self.gradient[-1])),
                          ITERATION,
                          self.accuracy[-1],
                          self.loss[-1],
                          self.learning_rate))

            t_after = process_time()
            self.cpu_time.append(t_after - t_before)

            if abs(np.linalg.norm(np.array(self.gradient[-1]))) < self.threshold:
                print("grad norm threshold: stopped iterations")
                break

            if  ITERATION > 2 and self.early_stopping():
                print("early_stopping: stopped iterations")
                break


class BCGD(Descent):
    def __init__(self, learning_rate=1e-5,
                 max_iterations=100, flag_nesterov_rand_block = True, learning_rate_strategy ='constant'):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.name = "Block_Descent"
        self.nesterov_rand_block = flag_nesterov_rand_block
        self.learning_rate_strategy = learning_rate_strategy

        self.gradient = []
        self.curr_rand_block= 0


    def _hessian_matrix(self):
        h_mat = np.copy(-self.weight_uu)
        for i in range(len(self.unlabeled_indices)):
            h_mat[i][i] += 2 * (np.sum(self.weight_lu[:, i]) + np.sum(self.weight_uu[:, i])) - self.weight_uu[i,i]
        return h_mat

    def _Larger_lipschitz_constant(self):
        Li = np.diag(self._hessian_matrix())
        return Li

    def _Lipschitz_constant(self):
        eig_vals, _ = np.linalg.eig(self._hessian_matrix())
        return max(eig_vals)

    def _learning_rate(self):

        if self.learning_rate_strategy == 'constant' or self.learning_rate_strategy == 0 :
            return self.learning_rate
        elif self.learning_rate_strategy == 'block_based' or self.learning_rate_strategy == 1:
            Li = self._Larger_lipschitz_constant()
            return 1 / Li[self.curr_rand_block]
        elif self.learning_rate_strategy == 'lipschitz' or self.learning_rate_strategy == 2:
            L = self._Lipschitz_constant()
            return 1 / L
        #           elif self.learning_rate_strategy == 'lipschitz2' or self.learning_rate_strategy ==3:
        #                L = self._Lipschitz_constant() / len(self.unlabeled_indices) # L/b
        #                self.learning_rate = 1 / L


class Randomized_BCGD(BCGD):
    def __init__(self, learning_rate=1e-5,
                 max_iterations=100, flag_nesterov_rand_block = True, learning_rate_strategy ='constant'):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.name = "R_BCGD"
        self.nesterov_rand_block = flag_nesterov_rand_block
        self.learning_rate_strategy = learning_rate_strategy

        self.gradient = []
        self.curr_rand_block= 0

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
            self.loss.append(self.calculate_loss(self.y[self.labeled_indices],self.y[self.unlabeled_indices]))
            self.calculate_accuracy()

            # Choosing random block
            if self.nesterov_rand_block:
                Li = self._Larger_lipschitz_constant()
                probs_Nesterov = Li / np.sum(Li)
                self.curr_rand_block = np.random.choice(range(len(self.unlabeled_indices)), p=probs_Nesterov)
            else:
                self.curr_rand_block = np.random.randint(len(self.unlabeled_indices))

            # Choose learning rate
            self.learning_rate = self._learning_rate()

            # Calculate gradient with respect to i
            self.calculate_gradient(self.curr_rand_block)

            # Update the estimated y
            self.y[self.unlabeled_indices[self.curr_rand_block]] = self.y[self.unlabeled_indices[self.curr_rand_block]] - \
                                                         self.learning_rate * self.gradient[-1]

            print("iteration: {} --- accuracy: {:.3f} ---- loss: {:.3f} --- next_stepsize: {}"
                  .format(ITERATION,
                          self.accuracy[-1],
                          self.loss[-1],
                          self.learning_rate))

            t_after = process_time()
            self.cpu_time.append(t_after - t_before)

            if  ITERATION > 2 and self.early_stopping():
                print("early_stopping: stopped iterations")
                break

class GS_BCGD(BCGD):
    def __init__(self, learning_rate=1e-5,
                 max_iterations=100,flag_nesterov_rand_block = True, learning_rate_strategy ='constant'):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.name = "GS_BCGD"

        self.nesterov_rand_block = flag_nesterov_rand_block
        self.learning_rate_strategy = learning_rate_strategy

        self.gradient = []
        self.curr_rand_block= 0

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

    def optimize(self):

        stop_condition = False
        ITERATION = 0

        while ITERATION < self.max_iterations:

            t_before = process_time()
            ITERATION += 1

            # Compute objective function for estimated y
            self.loss.append(self.calculate_loss(self.y[self.labeled_indices],self.y[self.unlabeled_indices]))
            self.calculate_accuracy()

            # Choosing max gradient block
            grad, max_gradient_index = self.get_largest_gradient_index()
            self.gradient.append(grad)

            # Choose learning rate
            self.learning_rate = self._learning_rate()

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

            if  ITERATION > 2 and self.early_stopping():
                print("early_stopping: stopped iterations")
                break


# Data Creation Functions
def data_creation(total_samples,unlabelled_ratio):

    # set the seed for reproducibility - in order to affect data point generation as well
    np.random.seed(10)

    # generate random data points with 2 features and 2 labels
    x, y = make_blobs(n_samples=total_samples, n_features=2, centers=2, cluster_std=1,
                                random_state=10)
    y = 2 * y - 1

    # make %unlabelled_ratio of data points unlabeled
    num_unlabeled_samples = int(unlabelled_ratio * total_samples)

    # all_indices = labeled indices + unlabeled indices
    unlabeled_indices = np.random.choice(total_samples, size=num_unlabeled_samples, replace=False)
    labeled_indices = np.array(list(set(np.array(range(total_samples))) - set(unlabeled_indices)))

    weight_lu, weight_uu = create_similarity_matrices(x,labeled_indices,unlabeled_indices)

    return total_samples,unlabelled_ratio,x,y,unlabeled_indices,labeled_indices,weight_lu,weight_uu


def create_similarity_matrices(x,labeled_indices,unlabeled_indices):
    eps = 1e-8  # not to get 0 in denominator
    weight_lu = 1 / (euclidean_distances(x[labeled_indices], x[unlabeled_indices]) + eps)
    weight_uu = 1 / (euclidean_distances(x[unlabeled_indices], x[unlabeled_indices]) + eps)

    return weight_lu, weight_uu



if __name__ == '__main__':


    #Save the current time
    start_time = time.time()
    #gd = GradientDescent(total_samples=1000, unlabelled_ratio=0.9,
     #                         learning_rate=0.0001, threshold=0.0001, max_iterations=100)

    rbcgd = Randomized_BCGD(learning_rate=0.001, max_iterations=1000,
                            learning_rate_strategy='block_based', flag_nesterov_rand_block=True)

    gd = Gradient_Descent( learning_rate=0.001, threshold=0.01,
                             max_iterations=1000, learning_rate_strategy='constant')

    gs = GS_BCGD( learning_rate=0.001,max_iterations=1000, learning_rate_strategy='constant')

    # data : tuple, x,y, labelled unlabelled indices, weight matrices
    data = data_creation(total_samples=1000,unlabelled_ratio=0.9)

    optimization_algorithms = [rbcgd,gs,gd] # add more/use only one
    for optim_alg in optimization_algorithms:
        print(f"{optim_alg.name} Start")
        start_time = time.time()
        optim_alg.load_data(*data)
        #optim_alg.create_data()
        #optim_alg.create_similarity_matrices()
        optim_alg.optimize()
        optim_alg.plot_points()
        optim_alg.plot_loss(save_plot=False)
        optim_alg.plot_accuracy(save_plot=False)
        optim_alg.plot_cpu_time(save_plot=False)
        #optim_alg.save_output()
        elapsed_time = time.time() - start_time
        print(f"Time Spent:{elapsed_time}")
        print(f"*"*100)

    print("end")
