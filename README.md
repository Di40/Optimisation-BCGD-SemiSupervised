# Gradient Descent and Block Coordinate Gradient Descent for Semi supervised learning

To read in detail about this project, please refer to the written [report](https://github.com/Di40/Optimisation-BCGD-SemiSupervised/blob/main/Report.pdf).

## Table of contents
* [Project description](#project-description)
* [Code structure](#code-structure)
* [Team members](#team-members)

## Project description

In this study, various optimization algorithms including standard **Gradient Descent**, **Block Coordinate Gradient Descent (BCGD) with Randomized rule** and **Gauss Southwel rule**, are used to address a **semi-supervised learning** problem. To accomplish this, a set of random points and labels are initially generated, with most of the labels being later discarded. The accuracy of data point classification and the loss function are computed for each algorithm. Additionally, an analysis of the loss function, number of iterations and CPU time is conducted. Finally, the effectiveness of the algorithms is evaluated using a real-world dataset.

*Generated clusters of points with labeled and unlabeled points:*

![image](https://github.com/Di40/Optimisation-BCGD-SemiSupervised/assets/57565142/9faa2180-e5c3-44bd-9592-879430d0d29d)

For each of the algorithms, different learning rate strategies are employed, including: constant learning rate, 1/Lipschitz constant, 1/Li and Armijo Rule.

### Results

We tested the three chosen algorithms on a [Credit card defaulter](https://www.kaggle.com/datasets/d4rklucif3r/defaulter) data set that we obtained from Kaggle. We decided to use the features balance and income. Because of resources limitations, we performed the tests on a subset of points, chosen randomly. The original dataset has 5776 rows, but as per the training we decided to use 1000 of them. During training we used 10% as the ratio of the labeled points, but here we proved that even having a smaller portion of labeled points works well. In our case, we chose 3% as the number of labeled points. From the comparison between the three chosen models, we drew the following conclusions:
* Both the gradient descent and BCGD with Gauss Southwell rule managed to label all of the points correctly (100% accuracy).
* Randomized BCGD yielded 82% accuracy.
* GS BCGD needed most time to complete, which was expected. However Randomized BCGD took more time than the Gradient Descent, which was unexpected. One possible explanation would be that the calculation of
 the Hessian (for obtaining Li) is more costly than computing the full gradient.

*Results for the credit card defaulter dataset:* 

![image](https://github.com/Di40/Optimisation-BCGD-SemiSupervised/assets/57565142/7c1c5892-cc6a-4586-9896-d7f8e11afda4)

![image](https://github.com/Di40/Optimisation-BCGD-SemiSupervised/assets/57565142/54aa4ba1-43e9-48c2-977b-c98d24dc8a4e)


# Code structure
* [optimisation.py](https://github.com/Di40/Optimisation-BCGD-SemiSupervised/blob/main/optimization.py) contains the classes and methods for each of the three algorithms.
* [utils.py](https://github.com/Di40/Optimisation-BCGD-SemiSupervised/blob/main/utils.py) contains helper functions needed for generating artificial dataset, loading the real dataset and for plotting.
* [test.py](https://github.com/Di40/Optimisation-BCGD-SemiSupervised/blob/main/test.py) is used for testing different algorithms and learning rates.

## Team members:
- Dejan Dichoski
- [Marija Cveevska](https://github.com/marijacveevska)
- [Suleyman Erim](https://github.com/suleymanerim1)
