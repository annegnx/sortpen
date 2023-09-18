import numpy as np

from numpy.testing import assert_allclose
from numpy.linalg import norm

import scipy.optimize as opt

from sortpen.penalties import *
from skglm.penalties import L1, SLOPE, MCPenalty, L0_5
from skglm.utils.data import make_correlated_data

n_samples = 20
n_features = 10
n_tasks = 10
X, Y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, n_tasks=n_tasks, density=1.,
    random_state=0, rho=0.9)
y = Y[:, 0]

n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = alpha_max / 1000
alphas = (np.linspace(alpha/2, 4*alpha, n_samples)**2)[::-1]
tol = 1e-10


class TestPenalties():
    def test_slope(self):
        our_slopePen = SortedL1Penalty()
        skglm_slope = SLOPE(alphas)
        skglm_L1 = L1(alpha)

        assert_allclose(our_slopePen.penalty(y, alphas), skglm_slope.value(y))
        assert_allclose(our_slopePen.prox_1D(
            y[0], alpha), skglm_L1.prox_1d(y[0], 1, 0))

        res_scipy = opt.minimize(lambda t: 1/(2) * norm(t-y) **
                                 2 + skglm_slope.value(t),  x0=np.random.randn(n_samples))

        our_res = our_slopePen.prox(y, alphas)

        skglm_res = skglm_slope.prox_vec(y, 1)

        assert_allclose(our_res, res_scipy.x)

        assert_allclose(our_slopePen.prox(y, alphas),
                        skglm_slope.prox_vec(y, 1))


if __name__ == '__main__':
    tester = TestPenalties()
    tester.test_slope()
