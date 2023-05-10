from optimization import Gradient_Descent, Randomized_BCGD, GS_BCGD
from utils import real_data, data_creation, plot_curves, plot_bar_metrics, plot_bar_per_model
import time
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # data : tuple, x,y, labelled unlabelled indices, weight matrices
    #data = data_creation(total_samples=1000, unlabelled_ratio=0.9)
    data = real_data(unlabelled_ratio=0.9)

    gd1 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.0001)
    gd2 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='lipschitz')
    gd3 = Gradient_Descent(threshold=0.01, max_iterations=5000, learning_rate_strategy='armijo')
    # TODO: Compare gd1 vs gd2 vs gd3

    r_bcgd1 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.001)
    r_bcgd2 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='lipschitz')
    r_bcgd3 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='block_based')
    r_bcgd4 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='armijo')
    r_bcgd5 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='block_based', use_nesterov_probs=True)
    # TODO: Compare the following:
        # r_bcgd1 vs r_bcgd2 vs r_bcgd3 vs r_bcgd4
        # r_bcgd1 vs r_bcgd3 vs r_bcgd5

    gs_bcgd1 = GS_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.001)
    gs_bcgd2 = GS_BCGD(max_iterations=5000, learning_rate_strategy='lipschitz')
    gs_bcgd3 = GS_BCGD(max_iterations=5000, learning_rate_strategy='block_based')
    gs_bcgd4 = GS_BCGD(max_iterations=5000, learning_rate_strategy='armijo')
    gs_bcgd5 = GS_BCGD(max_iterations=5000, learning_rate_strategy='block_based', use_nesterov_probs=True)

    optimization_algorithms = [gd1, gd2, gd3, r_bcgd1, r_bcgd2, r_bcgd3, r_bcgd4, r_bcgd5,
                               gs_bcgd1, gs_bcgd2, gs_bcgd3, gs_bcgd4, gs_bcgd5]

    elapsed_time = []  # total time needed for current algorithm

    col_names = ["loss", "cpu_time", "accuracy"]
    df_results = pd.DataFrame(columns=col_names)

    for count, optim_alg in enumerate(optimization_algorithms):
        print(f"{optim_alg.name} Start")
        start_time = time.time()
        optim_alg.load_data(*data)
        optim_alg.optimize()
        optim_alg.plot_points()
        optim_alg.plot_loss(save_plot=False)
        optim_alg.plot_accuracy(save_plot=False)
        optim_alg.plot_cpu_time(save_plot=False)
        optim_alg.save_output()
        elapsed_time.append(time.time() - start_time)

        df_results.loc[count] = [optim_alg.loss, optim_alg.cpu_time, optim_alg.accuracy]

        if count==1:
            break

    print(f"Time Spent: {elapsed_time}")
    from tabulate import tabulate
    print(tabulate(df_results, headers='keys'))

    iteration_list = [len(sub_list) for sub_list in df_results["loss"]]
    iteration_list = [list(range(1, n+1)) for n in iteration_list]
    cpu_time_list = [np.cumsum(l) for l in df_results["cpu_time"].tolist()]

    # plot_curves(df_results["loss"].tolist(), iteration_list,
    #             "Iteration", "Loss", "Loss vs Iteration",
    #             df_results.index.tolist())

    # plot_curves(df_results["loss"].tolist(), cpu_time_list,
    #             "CPU Time", "Loss", "Loss vs CPU Time",
    #             df_results.index.tolist())

    # plot_curves(df_results["accuracy"].tolist(), iteration_list,
    #             "Iteration", "Accuracy", "Accuracy vs Iteration",
    #             df_results.index.tolist())

    #plot_bar_metrics(df_results)

    #plot_bar_per_model(df_results, "loss")
    #plot_bar_per_model(df_results, "accuracy")
    #plot_bar_per_model(df_results, "cpu_time")
    plot_bar_per_model(df_results, "iterations")
