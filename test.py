from optimization import GradientDescent, Randomized_BCGD, GS_BCGD
from utils import real_data, data_creation, plot_curves, plot_bar_metrics, plot_bar_per_model
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # data : tuple, x,y, labelled unlabelled indices, weight matrices
    # data = data_creation(total_samples=1000, unlabelled_ratio=0.9)
    data = real_data(unlabelled_ratio=0.9)

    #################################################

    # Compare the performance of GD with different fixed learning rates:

    # gd_fixed_1 = GradientDescent(threshold=0.01, max_iterations=5000,
    #                              learning_rate_strategy='constant', learning_rate=0.0005)
    # gd_fixed_1.name = 'LR Fixed ' + str(gd_fixed_1.learning_rate)
    # gd_fixed_2 = GradientDescent(threshold=0.01, max_iterations=5000,
    #                              learning_rate_strategy='constant', learning_rate=0.0001)
    # gd_fixed_2.name = 'LR Fixed ' + str(gd_fixed_2.learning_rate)
    # gd_fixed_3 = GradientDescent(threshold=0.01, max_iterations=5000,
    #                              learning_rate_strategy='constant', learning_rate=0.001)
    # gd_fixed_3.name = 'LR Fixed ' + str(gd_fixed_3.learning_rate)

    # gd_fixed_1 performed the best, so we keep it, and compare it to other learning rate approaches:

    # gd1 = GradientDescent(threshold=0.01, max_iterations=5000, learning_rate_strategy='constant',
    #                              learning_rate=0.0005)
    # gd1.name = 'LR Fixed ' + str(gd1.learning_rate)
    # gd2 = GradientDescent(threshold=0.01, max_iterations=5000, learning_rate_strategy='lipschitz')
    # gd2.name = 'LR 1/Lipschitz'
    # gd3 = GradientDescent(threshold=0.01, max_iterations=5000, learning_rate_strategy='armijo')
    # gd3.name = 'LR ArmijoRule'

    # After testing, gd2, i.e., LR 1/Lipschitz performed the best. So we decide to keep it for the final analysis.

    #################################################

    # Compare the performance of Randomized BCGD with different fixed learning rates:

    # r_bcgd_fixed_1 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.0001)
    # r_bcgd_fixed_1.name = 'LR Fixed ' + str(r_bcgd_fixed_1.learning_rate)
    # r_bcgd_fixed_2 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.0005)
    # r_bcgd_fixed_2.name = 'LR Fixed ' + str(r_bcgd_fixed_2.learning_rate)
    # r_bcgd_fixed_3 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.001)
    # r_bcgd_fixed_3.name = 'LR Fixed ' + str(r_bcgd_fixed_3.learning_rate)

    # r_bcgd_fixed_2 performed the best, so we keep it, and compare it to other learning rate approaches:

    # r_bcgd1 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='constant', learning_rate=0.001)
    # r_bcgd1.name = 'LR Fixed ' + str(r_bcgd1.learning_rate)
    # r_bcgd2 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='lipschitz')
    # r_bcgd2.name = 'LR 1/Lipschitz'
    # r_bcgd3 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='block_based')
    # r_bcgd3.name = 'LR 1/Li'
    # r_bcgd4 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='armijo')
    # r_bcgd4.name = 'LR ArmijoRule'
    # r_bcgd5 = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='block_based', use_nesterov_probs=True)
    # r_bcgd5.name = 'Nesterov sampling, LR 1/Li'

    # After testing r_bcgd3, i.e., LR 1/Li performed the best. So we decide to keep it for the final analysis.

    #################################################

    # Compare the performance of Randomized BCGD with different fixed learning rates:

    # gs_bcgd_fixed_1 = GS_BCGD(max_iterations=1000, learning_rate_strategy='constant', learning_rate=0.0001)
    # gs_bcgd_fixed_1.name = 'LR Fixed ' + str(gs_bcgd_fixed_1.learning_rate)
    # gs_bcgd_fixed_2 = GS_BCGD(max_iterations=1000, learning_rate_strategy='constant', learning_rate=0.0005)
    # gs_bcgd_fixed_2.name = 'LR Fixed ' + str(gs_bcgd_fixed_2.learning_rate)
    # gs_bcgd_fixed_3 = GS_BCGD(max_iterations=1000, learning_rate_strategy='constant', learning_rate=0.001)
    # gs_bcgd_fixed_3.name = 'LR Fixed ' + str(gs_bcgd_fixed_2.learning_rate)

    # In the analysis 1000 iterations were used, because GS_BCGD is much slower (we compute the full gradient).
    # However, accuracy was constantly increasing, so more iterations (5000) would give better results.

    # gd_fixed_2 performed the best, so we keep it, and compare it to other learning rate approaches:

    # gs_bcgd1 = GS_BCGD(max_iterations=1000, learning_rate_strategy='constant', learning_rate=0.0005)
    # gs_bcgd1.name = 'LR Fixed'
    # gs_bcgd2 = GS_BCGD(max_iterations=1000, learning_rate_strategy='lipschitz')
    # gs_bcgd2.name = 'LR 1/Lipschitz'
    # # gs_bcgd3 = GS_BCGD(max_iterations=1000, learning_rate_strategy='armijo')
    # # gs_bcgd3.name = 'LR ArmijoRule'
    # gs_bcgd3 = GS_BCGD(max_iterations=1000, learning_rate_strategy='block_based')  # Li
    # gs_bcgd3.name = 'LR 1/Li'
    # gs_bcgd4 = GS_BCGD(max_iterations=1000, use_Li_for_block_selection=True, learning_rate_strategy='block_based')
    # gs_bcgd4.name = 'LR 1/Li, |grad/Li|'
    # gs_bcgd5 = GS_BCGD(max_iterations=1000, use_Li_for_block_selection=True, learning_rate_strategy='block_based')
    # gs_bcgd5.name = 'LR 1/L, |grad/Li|'

    # After testing gs_bcgd4, i.e., block = argmax(|grad/Li|) with LR=1/Li, performed the best.
    # So we decide to keep it for the final analysis.

    #################################################

    # We keep one configuration for each method (GD, R_BCGD and GS_BCGD) and compare their performances on the dummy
    # dataset and on a real dataset. Also, we increase the number of iterations to enable for more training time.

    gd = GradientDescent(threshold=0.01, max_iterations=5000, learning_rate_strategy='lipschitz')
    gd.name = gd.name + ' LR 1/Lipschitz'
    r_bcgd = Randomized_BCGD(max_iterations=5000, learning_rate_strategy='block_based')
    r_bcgd.name = r_bcgd.name + ' LR 1/Li'
    gs_bcgd = GS_BCGD(max_iterations=5000, use_Li_for_block_selection=True, learning_rate_strategy='block_based')  # Li
    gs_bcgd.name = gs_bcgd.name + ' LR 1/Li, |grad/Li|'

    # Enter in the list the algorithms that you are testing:
    optimization_algorithms = [gd, r_bcgd, gs_bcgd]

    col_names = ["optim_alg", "loss", "cpu_time", "accuracy"]
    df_results = pd.DataFrame(columns=col_names)

    plotted = False

    for count, optim_alg in enumerate(optimization_algorithms):
        print(f"{optim_alg.name}")
        optim_alg.load_data(*data)
        optim_alg.optimize()
        # if not plotted:
        #     optim_alg.plot_points()
        #     plotted = True
        optim_alg.plot_loss(save_plot=False)
        optim_alg.plot_accuracy(save_plot=False)
        optim_alg.plot_cpu_time(save_plot=False)
        # optim_alg.save_output()
        df_results.loc[count] = [optim_alg.name, optim_alg.loss, optim_alg.cpu_time, optim_alg.accuracy]

    df_results.to_csv('optimization_results.csv', index=False)

    legend_list    = df_results["optim_alg"].tolist()
    iteration_list = [len(sub_list) for sub_list in df_results["loss"]]
    iteration_list = [list(range(1, n+1)) for n in iteration_list]
    cpu_time_list  = [np.cumsum(cpu) for cpu in df_results["cpu_time"].tolist()]

    plot_curves(df_results["loss"].tolist(), iteration_list,
                "Iteration", "Loss", "Loss vs Iteration", legend_list)
    plot_curves(df_results["loss"].tolist(), cpu_time_list,
                "CPU Time", "Loss", "Loss vs CPU Time", legend_list)
    plot_curves(df_results["accuracy"].tolist(), iteration_list,
                "Iteration", "Accuracy", "Accuracy vs Iteration", legend_list)
    plot_curves(df_results["accuracy"].tolist(), cpu_time_list,
                "CPU Time", "Accuracy", "Accuracy vs CPU Time", legend_list)

    plot_bar_metrics(df_results)

    plot_bar_per_model(df_results, "loss")
    plot_bar_per_model(df_results, "accuracy")
    plot_bar_per_model(df_results, "cpu_time")
    plot_bar_per_model(df_results, "iterations")
