import numpy as np
from time import time
from numpy.testing import assert_allclose
from sklearn.isotonic import isotonic_regression
from scipy.optimize import minimize, LinearConstraint

from sortpen.iso_prox import *


def lca_mcp_sorted(x, lmbda, gamma):
    return ((lmbda * x) * (x <= lmbda * gamma).astype(int) + (lmbda**2 * gamma/2. + x**2/(2*gamma)) * (x > lmbda * gamma).astype(int)).sum()


def mcp_sorted(x, lmbda, gamma):
    return ((lmbda * x - x**2/(2*gamma)) * (x <= lmbda * gamma).astype(int) + (lmbda**2 * gamma/2.) * (x > lmbda * gamma).astype(int)).sum()


def objective_lca_smcp(x, v, lmb, gamma):
    return 0.5 * ((v-x)**2).sum() + lca_mcp_sorted(x, lmb, gamma)


def objective_smcp(x, v, lmb, gamma):
    return 0.5 * ((v-x)**2).sum() + mcp_sorted(x, lmb, gamma)


class TestIsoProx():
    def test_iso_prox_regression_against_sklearn(self):
        n = 1000
        # v = np.array([0,  2,  1, 13,  8, 17, 18, 16,  6,  6]).astype(float)
        # lmb = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 1]).astype(float)
        v = np.random.randint(0, 5, n).astype(float)
        lmb = np.random.randint(0, 3, n).astype(float)**2
        iso_ope = IsoProxSlope()
        result = iso_ope(v, lmb)
        result_iso_baseline = isotonic_regression(v-lmb, increasing=False)
        assert_allclose(result, result_iso_baseline)

    def test_iso_prox_regression_against_optim(self):
        n = 100
        constraints_matrix = np.eye(n)
        constraints_matrix[-1, -1] = 0
        indices = np.arange(n-1)
        indices_shift = indices + 1
        constraints_matrix[indices, indices_shift] = -1
        cons = LinearConstraint(constraints_matrix, lb=0)
        # v = np.array([0,  2,  1, 13,  8, 17, 18, 16,  6,  6]).astype(float)
        # lmb = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 1]).astype(float)
        v = np.random.randint(0, 5, n).astype(float)
        lmb = np.random.randint(0, 3, n).astype(float)**2
        iso_ope = IsoProxSlope()
        result = iso_ope(v, lmb)
        result_iso_baseline = minimize(lambda x: (
            ((v-lmb)-x)**2).sum(), x0=(v-lmb), constraints=cons).x

        assert_allclose(result, result_iso_baseline, atol=1e-5)

    def test_iso_prox_lca_against_optim(self):
        n = 100
        constraints_matrix = np.eye(n)
        constraints_matrix[-1, -1] = 0
        indices = np.arange(n-1)
        indices_shift = indices + 1
        constraints_matrix[indices, indices_shift] = -1
        cons = LinearConstraint(constraints_matrix, lb=0)
        gamma = 3
        # v = np.array([0,  2,  1, 13,  8, 17, 18, 16,  6,  6]).astype(float)
        # v.sort()
        # v = v[::-1]
        # lmb = np.array([6, 3, 3, 3, 3, 0, 0, 0, 0, 0]).astype(float)
        v = np.random.randint(0, 5, n).astype(float)
        v.sort()
        v = v[::-1]
        lmb = np.random.randint(0, 3, n).astype(float)**2
        lmb.sort()
        lmb = lmb[::-1]
        iso_ope = IsoProxLCA()
        t1 = time()
        result_explicit = iso_ope(v, lmb, gamma=gamma)
        print("Explicit prox", time()-t1)
        iso_ope_implicit = IsoProxLCAImplicit()
        t2 = time()
        result_implicit = iso_ope_implicit(v, lmb, gamma=gamma)
        print("Implicit prox", time()-t2)
        t3 = time()
        result_iso_baseline = minimize(lambda x:
                                       0.5 * ((v-x)**2).sum() + lca_mcp_sorted(x, lmb, gamma), x0=np.minimum(v - lmb, v / (1+1/gamma)), constraints=cons).x
        print("Scipy minimization", time()-t3)

        objective_prox = objective_lca_smcp(result_explicit, v, lmb, gamma)
        objective_prox_implicit = objective_lca_smcp(
            result_implicit, v, lmb, gamma)
        objective_baseline = objective_lca_smcp(
            result_iso_baseline, v, lmb, gamma)

        assert_allclose(result_explicit, result_iso_baseline, atol=1e-3)
        assert_allclose(result_implicit, result_iso_baseline, atol=1e-3)
        assert_allclose(objective_prox, objective_baseline, atol=1e-3)
        assert_allclose(objective_prox_implicit, objective_baseline, atol=1e-3)

    def test_iso_prox_direct_smcp_against_optim(self):
        n = 100
        constraints_matrix = np.eye(n)
        constraints_matrix[-1, -1] = 0
        indices = np.arange(n-1)
        indices_shift = indices + 1
        constraints_matrix[indices, indices_shift] = -1
        cons = LinearConstraint(constraints_matrix, lb=0)
        gamma = 3
        # v = np.array([0,  2,  1, 13,  8, 17, 18, 16,  6,  6]).astype(float)
        # v.sort()
        # v = v[::-1]
        # lmb = np.array([6, 3, 3, 3, 3, 0, 0, 0, 0, 0]).astype(float)
        v = np.random.randint(0, 5, n).astype(float)
        v.sort()
        v = v[::-1]
        lmb = np.random.randint(0, 3, n).astype(float)**2
        lmb.sort()
        lmb = lmb[::-1]
        iso_ope = IsoProxSMCP()
        t1 = time()
        result_explicit = iso_ope(v, lmb, gamma=gamma)
        print("Explicit prox", time()-t1)
        iso_ope_implicit = IsoProxSMCPImplicit()
        t2 = time()
        result_implicit = iso_ope_implicit(v, lmb, gamma=gamma)
        print("Implicit prox", time()-t2)
        t3 = time()
        result_iso_baseline = minimize(lambda x:
                                       0.5 * ((v-x)**2).sum() + mcp_sorted(x, lmb, gamma), x0=np.minimum(v - lmb, v / (1+1/gamma)), constraints=cons).x
        print("Scipy minimization", time()-t3)
        objective_prox = objective_smcp(result_explicit, v, lmb, gamma)
        objective_prox_implicit = objective_smcp(
            result_implicit, v, lmb, gamma)
        objective_baseline = objective_smcp(result_iso_baseline, v, lmb, gamma)
        assert_allclose(
            result_explicit, result_iso_baseline, rtol=1e-1)
        assert_allclose(
            result_implicit, result_iso_baseline, rtol=1e-1)
        assert_allclose(
            objective_prox, objective_baseline, atol=1e-1)
        assert_allclose(
            objective_prox_implicit, objective_baseline, atol=1e-1)


if __name__ == '__main__':
    pass
