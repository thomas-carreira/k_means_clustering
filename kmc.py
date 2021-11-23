import numpy as np
import matplotlib.pyplot as plt
import random


def init_u_and_r(X, k):
    # X is a matrix of size nx2
    # k is the number of clusters
    min_i = 0
    max_i = len(X)-1
    rand = random.sample(range(min_i, max_i), k)
    u = X[rand]
    r, loss = get_membership_and_loss(X, u)

    return u, r


def get_membership_and_loss(X, u):
    # X is a matrix of size nx2
    # u has the mean for each cluster and has a dimension of kx2

    r = np.zeros([len(X), len(u)])
    loss = 0

    for n in range(len(X)):
        best_d = 1000
        best_k = 0

        for k in range(len(u)):
            d = np.sqrt(np.sum((X[n] - u[k])**2))
            if d < best_d:
                best_d = d
                best_k = k

        r[n, best_k] = 1
        # Je multiplie directement par 1 car best_k correspond à la bonne classe
        loss += 1 * np.sqrt(np.sum((X[n] - u[best_k]) ** 2))

    return r, loss


def optimisation(X, r, u):
    # X is a matrix of size nx2
    # r is a matrix of size nxk and has only 1 and 0
    # u has the mean for each cluster and has a dimension of kx2

    new_u = np.zeros(u.shape)

    for k in range(len(u)):
        divider = np.sum(r[:, k])
        if divider == 0:
            new_u[k] = u[k]
        else:
            new_u[k] = r[:, k].dot(X) / divider

    return new_u


def train(X, k, max_iter, display_graph=False):
    # X is a matrix of size nx2
    # k is the number of clusters
    # max_iter number of iteration max

    loss_hist = []
    nb_times_loss_is_equal = 0
    when_to_stop = 2

    u, r = init_u_and_r(X, k)

    for i in range(max_iter):
        u = optimisation(X, r, u)
        r, loss = get_membership_and_loss(X, u)
        loss_hist.append(loss)

        if display_graph: plot_k_means(X, r, u, k, loss)
        if i > 0 and loss == loss_hist[i - 1]:
            if nb_times_loss_is_equal == when_to_stop:
                break
            nb_times_loss_is_equal += 1

    print("Loss with k =", k ,"after", i, "itération =", loss_hist[i])
    return loss_hist


def plot_k_means(X, r, u, k, loss):
    # X is a matrix of size nx2
    # r is a matrix of size nxk and has only 1 and 0
    # u has the mean for each cluster and has a dimension of kx2
    # k is the number of clusters
    # loss compared to u and r

    for i in range(k):
        X_by_cluster = X[r[:, i] == 1]
        plt.plot(X_by_cluster[:, 0], X_by_cluster[:, 1], 'o')

    plt.plot(u[:, 0], u[:, 1], 'ko')
    plt.title("Loss = " + str(loss))
    plt.show()
