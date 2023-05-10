import main
import time

#Save the current time
    start_time = time.time()

    gd1 = Gradient_Descent(total_samples=5000, unlabelled_ratio=0.9,
                          threshold=0.01, max_iterations=5000,
                           learning_rate_strategy='constant', learning_rate=0.0001)
    gd2 = Gradient_Descent(total_samples=5000, unlabelled_ratio=0.9,
                           threshold=0.01, max_iterations=5000,
                           learning_rate_strategy='lipschitz')
    gd3 = Gradient_Descent(total_samples=5000, unlabelled_ratio=0.9,
                           threshold=0.01, max_iterations=5000,
                           learning_rate_strategy='armijo')

    #TODO: Compare gd1 vs gd2 vs gd3

    r_bcgd1 = Randomized_BCGD(total_samples=5000, unlabelled_ratio=0.9, learning_rate=0.001, max_iterations=5000,
                             learning_rate_strategy='constant')
    r_bcgd2 = Randomized_BCGD(total_samples=5000, unlabelled_ratio=0.9,
                              max_iterations=5000, learning_rate_strategy='lipschitz')
    r_bcgd3 = Randomized_BCGD(total_samples=5000, unlabelled_ratio=0.9, max_iterations=5000,
                              learning_rate_strategy='block_based')
    r_bcgd4 = Randomized_BCGD(total_samples=5000, unlabelled_ratio=0.9, max_iterations=5000,
                              learning_rate_strategy='armijo')
    r_bcgd5 = Randomized_BCGD(total_samples=5000, unlabelled_ratio=0.9, max_iterations=5000,
                              learning_rate_strategy='block_based', flag_nesterov_rand_block=True)

    #TODO: Compare the following:
    # r_bcgd1 vs r_bcgd2 vs r_bcgd3 vs r_bcgd4
    # r_bcgd1 vs r_bcgd3 vs r_bcgd5

    gs_bcgd1 = GS_BCGD(total_samples=5000, unlabelled_ratio=0.9, learning_rate=0.001, max_iterations=5000,
                       learning_rate_strategy='constant')
    gs_bcgd2 = GS_BCGD(total_samples=5000, unlabelled_ratio=0.9,
                       max_iterations=5000, learning_rate_strategy='lipschitz')
    gs_bcgd3 = GS_BCGD(total_samples=5000, unlabelled_ratio=0.9, max_iterations=5000,
                       learning_rate_strategy='block_based')
    gs_bcgd4 = GS_BCGD(total_samples=5000, unlabelled_ratio=0.9, max_iterations=5000,
                       learning_rate_strategy='armijo')
    gs_bcgd5 = GS_BCGD(total_samples=5000, unlabelled_ratio=0.9, max_iterations=5000,
                       learning_rate_strategy='block_based', flag_nesterov_rand_block=True)

    optimization_algorithms = [gd1, gd2, gd3, r_bcgd1, r_bcgd2, r_bcgd3, r_bcgd4, r_bcgd5,
                               gs_bcgd1, gs_bcgd2, gs_bcgd3, gs_bcgd4, gs_bcgd5]

    elapsed_time = [] # total time needed for current algorithm
    optim_alg_loss_list = []
    optim_alg_cpu_list = []
    optim_alg_acc_list = []
    i=0
    for optim_alg in optimization_algorithms:
        print(f"{optim_alg.name} Start")
        start_time = time.time()
        optim_alg.create_data()
        optim_alg.create_similarity_matrices()
        optim_alg.optimize()
        optim_alg.plot_points()
        optim_alg.plot_loss(save_plot=False)
        optim_alg.plot_accuracy(save_plot=False)
        optim_alg.plot_cpu_time(save_plot=False)
        optim_alg.save_output()

        elapsed_time.append(time.time() - start_time)

        optim_alg_loss_list.append(optim_alg.loss)
        optim_alg_cpu_list.append(optim_alg.cpu_time)
        optim_alg_acc_list.append(optim_alg.accuracy)

        print(f"*"*100)
        if i==2:
            break
    print(f"Time Spent: {elapsed_time}")
