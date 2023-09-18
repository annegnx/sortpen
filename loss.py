import numpy as np


def reg_loss(X, y, beta):
    return 0.5 * ((y - X @ beta)**2).sum()


def grad_reg_loss(X,  beta, y):
    return X.T @ (X @ beta - y) / len(y)


def grad_reg_loss_without_norm(X, y, beta):
    return X.T @ (X @ beta - y)
