from optimization import Gradient_Descent, Randomized_BCGD, GS_BCGD
from utils import real_data, data_creation, plot_curves, plot_bar_metrics, plot_bar_per_model
import time
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # data : tuple, x,y, labelled unlabelled indices, weight matrices
    data = data_creation(total_samples=1000, unlabelled_ratio=0.9)
    #data = real_data(unlabelled_ratio=0.9)

    #################################################

    # TODO: Marija - Pick best of these
    gd_fixed_1 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.0001)
    gd_fixed_1.name = 'LR Fixed ' + str(gd_fixed_1.learning_rate)
    gd_fixed_2 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.0005)
    gd_fixed_2.name = 'LR Fixed ' + str(gd_fixed_2.learning_rate)
    gd_fixed_3 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.001)
    gd_fixed_3.name = 'LR Fixed ' + str(gd_fixed_3.learning_rate)
    gd_fixed_4 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.005)
    gd_fixed_4.name = 'LR Fixed ' + str(gd_fixed_4.learning_rate)
    gd_fixed_5 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.01)
    gd_fixed_5.name = 'LR Fixed ' + str(gd_fixed_5.learning_rate)

    # If gd_fixed_5 is the best:
    # gd1 = gd_fixed5

    gd2 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='lipschitz')
    gd2.name = 'LR 1/Lipschitz'
    gd3 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='armijo')
    gd3.name = 'LR ArmijoRule'
    # TODO: Marija - Compare gd1 vs gd2 vs gd3 -> Pick best Gradient Method

    #################################################

    # TODO: Suleyman - Pick best of these
    r_bcgd_fixed_1 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.0001)
    r_bcgd_fixed_1.name = 'LR Fixed ' + str(r_bcgd_fixed_1.learning_rate)
    r_bcgd_fixed_2 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.0005)
    r_bcgd_fixed_2.name = 'LR Fixed ' + str(r_bcgd_fixed_2.learning_rate)
    r_bcgd_fixed_3 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.001)
    r_bcgd_fixed_3.name = 'LR Fixed ' + str(r_bcgd_fixed_3.learning_rate)
    r_bcgd_fixed_4 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.005)
    r_bcgd_fixed_4.name = 'LR Fixed ' + str(r_bcgd_fixed_4.learning_rate)
    r_bcgd_fixed_5 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.01)
    r_bcgd_fixed_5.name = 'LR Fixed ' + str(r_bcgd_fixed_5.learning_rate)

    # If r_bcgd_fixed_5 is the best:
    # r_bcgd1 = r_bcgd_fixed_5
    r_bcgd2 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='lipschitz')
    r_bcgd2.name = 'LR 1/Lipschitz'
    r_bcgd3 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='block_based')
    r_bcgd3.name = 'LR 1/Li'
    r_bcgd4 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='armijo')
    r_bcgd4.name = 'LR ArmijoRule'
    r_bcgd5 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='block_based', use_nesterov_probs=True)
    r_bcgd5.name = 'Nesterov sampling, LR 1/Li'
    # TODO: Suleyman - Compare the following:
        # r_bcgd1 vs r_bcgd2 vs r_bcgd3 vs r_bcgd4
        # r_bcgd1 vs r_bcgd3 vs r_bcgd5

    #################################################

    # TODO: Dejan - Pick best of these

    gs_bcgd_fixed_1 = GS_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.0001)
    gs_bcgd_fixed_1.name = 'LR Fixed' + str(gs_bcgd_fixed_1.learning_rate)
    gs_bcgd_fixed_2 = GS_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.0005)
    gs_bcgd_fixed_2.name = 'LR Fixed' + str(gs_bcgd_fixed_2.learning_rate)
    gs_bcgd_fixed_3 = GS_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.001)
    gs_bcgd_fixed_3.name = 'LR Fixed' + str(gs_bcgd_fixed_3.learning_rate)
    gs_bcgd_fixed_4 = GS_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.05)
    gs_bcgd_fixed_4.name = 'LR Fixed' + str(gs_bcgd_fixed_4.learning_rate)
    gs_bcgd_fixed_5 = GS_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.01)
    gs_bcgd_fixed_5.name = 'LR Fixed' + str(gs_bcgd_fixed_5.learning_rate)

    # If gs_bcgd_fixed_5 is the best:
    # gs_bcgd1 = gs_bcgd_fixed_5
    gs_bcgd2 = GS_BCGD(max_iterations=5000, learning_rate_strategy='lipschitz')
    gs_bcgd2.name = 'LR 1/Lipschitz'
    gs_bcgd3 = GS_BCGD(max_iterations=5000, learning_rate_strategy='block_based')
    gs_bcgd3.name = 'LR 1/Li'
    gs_bcgd4 = GS_BCGD(max_iterations=5000, learning_rate_strategy='armijo')
    gs_bcgd4.name = 'LR ArmijoRule'
    # TODO: Dejan - Compare the following:
        # gs_bcgd1 vs gs_bcgd2 vs gs_bcgd3 vs gs_bcgd4


    # optimization_algorithms = [gd1, gd2, gd3, r_bcgd1, r_bcgd2, r_bcgd3, r_bcgd4, r_bcgd5,
    #                            gs_bcgd1, gs_bcgd2, gs_bcgd3, gs_bcgd4, gs_bcgd5]

    optimization_algorithms = [gd2] # TODO: Enter here the algorithms that you are testing (as shown above)

    col_names = ["optim_alg", "loss", "cpu_time", "accuracy"]
    df_results = pd.DataFrame(columns=col_names)
    legend_list = []

    for count, optim_alg in enumerate(optimization_algorithms):
        print(f"{optim_alg.name}")
        optim_alg.load_data(*data)
        optim_alg.optimize()
        # optim_alg.plot_points()
        # optim_alg.plot_loss(save_plot=False)
        # optim_alg.plot_accuracy(save_plot=False)
        # optim_alg.plot_cpu_time(save_plot=False)
        # optim_alg.save_output()
        df_results.loc[count] = [optim_alg.name, optim_alg.loss, optim_alg.cpu_time, optim_alg.accuracy]

    df_results.to_csv('optimization_results.csv', index=False)

    legend_list    = df_results["optim_alg"].tolist()
    iteration_list = [len(sub_list) for sub_list in df_results["loss"]]
    iteration_list = [list(range(1, n+1)) for n in iteration_list]
    cpu_time_list  = [np.cumsum(l) for l in df_results["cpu_time"].tolist()]

    #TODO: Uncomment/Comment lines as you need them for testing
    plot_curves(df_results["loss"].tolist(), iteration_list,
                "Iteration", "Loss", "Loss vs Iteration", legend_list)

    plot_curves(df_results["loss"].tolist(), cpu_time_list,
                "CPU Time", "Loss", "Loss vs CPU Time", legend_list)

    plot_curves(df_results["accuracy"].tolist(), iteration_list,
                "Iteration", "Accuracy", "Accuracy vs Iteration", legend_list)

    # plot_bar_metrics(df_results)

    # plot_bar_per_model(df_results, "loss")
    # plot_bar_per_model(df_results, "accuracy")
    # plot_bar_per_model(df_results, "cpu_time")
    # plot_bar_per_model(df_results, "iterations")
