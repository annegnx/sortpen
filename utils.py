import numpy as np
from scipy import stats


def dual_norm_slope(X, theta, alphas):
    """Dual slope norm of X.T @ theta"""
    Xtheta = np.sort(np.abs(X.T @ theta))[::-1]
    taus = 1 / np.cumsum(alphas)
    return np.max(np.cumsum(Xtheta) * taus)


def lambda_sequence(X, y, fit_intercept, reg=0.1, q=0.1):
    """Generates the BH-type lambda sequence"""
    n, p = X.shape

    randnorm = stats.norm(loc=0, scale=1)
    lambdas = randnorm.ppf(1 - np.arange(1, p + 1) * q / (2 * p))
    lambda_max = dual_norm_slope(
        X, (y - np.mean(y) * fit_intercept) / n, lambdas)

    return lambda_max * lambdas * reg


def lambda_univ(sigma, n, p):
    return sigma * np.sqrt((2/n) * np.log(p))


def gamma_univ(X):
    n = X.shape[0]
    prod = (X.T @ X)
    diag = np.diag(prod)
    max_ = np.amax(np.abs(prod - np.diag(diag))) * 2 / n**2
    return 1 / (1-max_)


def mse(predictions, true_beta):
    return np.linalg.norm(predictions - true_beta, axis=1, ord=2)


def standardization(X):
    mean = X.mean(axis=0)
    X = X - mean
    norm = np.linalg.norm(X, ord=2, axis=0)
    X /= norm
    return X


X_t = np.array([[1, -8, 3, 1], [4, 1, 6, 12], [-33, 1, 6, 1]])
res = standardization(X_t)
