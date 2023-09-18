import numpy as np
from skglm.utils.data import make_correlated_data
from skglm.estimators import Lasso, MCPRegression
from skglm import GeneralizedLinearEstimator
from skglm.solvers import FISTA
from skglm.penalties import SLOPE
from slope.solvers import pgd_slope
from slope.utils import lambda_sequence
from numpy.testing import assert_allclose

from sortpen.solvers import *
from sortpen.loss import grad_reg_loss


def mse(predictions, true_beta):
    return np.linalg.norm(predictions - true_beta, axis=1, ord=2)


n_samples = 20
n_features = 10
n_tasks = 10
X, Y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, n_tasks=n_tasks, density=0.5,
    random_state=0)
y = Y[:, 0]

n_samples, n_features = X.shape
alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / n_samples
alpha = alpha_max / 1000

tol = 1e-10


class TestIsoProx():
    def test_slope_lasso(self):
        # compare solution with skglm Lasso
        alphas = np.full(n_features, alpha)
        our_lasso = Slope(grad_reg_loss, alphas, max_steps=5000)

        our_solution, iters = our_lasso.fit(X, y)

        lasso = Lasso(alpha, fit_intercept=False, tol=tol).fit(X, y)

        assert_allclose(lasso.coef_, our_solution, rtol=1e-5)

    def test_slope(self):
        # compare solutions with skglm (Fista, Slope) and `pyslope`: https://github.com/jolars/pyslope
        q = 0.1
        alphas = lambda_sequence(
            X, y, fit_intercept=False, reg=alpha / alpha_max, q=q)
        pyslope_out = pgd_slope(
            X, y, alphas, fit_intercept=False, max_it=1000, gap_tol=tol)
        our_slope = Slope(grad_reg_loss, alphas, max_steps=1000)
        our_solution, iters = our_slope.fit(X, y)
        skglm_solution = GeneralizedLinearEstimator(
            penalty=SLOPE(alphas),
            solver=FISTA(max_iter=1000, tol=tol, opt_strategy="fixpoint"),).fit(X, y)
        assert_allclose(
            pyslope_out["beta"], our_solution, rtol=1e-5)
        assert_allclose(
            skglm_solution.coef_, our_solution, rtol=1e-5)

    def test_mcp(self):
        # compare ith skglm MCPRegression
        alphas = np.full(n_features, alpha)
        our_slope = SlopeMCP_LCA(
            grad_reg_loss, 4, alphas, max_steps=1000, max_iters=20)
        our_solution, iters = our_slope.fit(X, y)
        our_direct_slope = SlopeMCP_Direct(
            grad_reg_loss, 4, alphas, max_steps=1000)
        our_direct_solution, iters = our_direct_slope.fit(X, y)

        skglm_solution = MCPRegression(
            alpha, 4, max_iter=1000, fit_intercept=False, ws_strategy="fixpoint").fit(X, y)
        assert_allclose(
            skglm_solution.coef_, our_solution, rtol=1e-2)
        assert_allclose(
            skglm_solution.coef_, our_direct_solution, rtol=1e-2)


if __name__ == '__main__':
    pass
