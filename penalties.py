import numpy as np
from scipy.optimize import bisect
from sortpen.iso_prox import SortedProx


class SortedPenaltyBase:
    def __init__(self) -> None:
        self.proxOpe = None

    def penalty(self, x, lmbdas, gamma=None):
        pass

    def prox_objective(self, x, y, lmbdas, eta=1, gamma=None):
        lmbdas = np.sort(lmbdas)[::-1]
        return (1./(2. * eta)) * np.linalg.norm(x-y)**2 + self.penalty(x, lmbdas, gamma)

    def prox_1D(self, y, lmbda, eta=1, gamma=None):
        pass

    def prox(self, y, lmbdas, eta=1, gamma=None):
        return self.proxOpe(y, lmbdas, eta, gamma)


class SortedL1Penalty(SortedPenaltyBase):
    def __init__(self):
        self.proxOpe = SortedProx(name_of_prox="slope")
        name = "Sorted L1"

    def penalty(self, x, lmbdas, gamma=None):
        abs_x_sorted = np.sort(np.abs(x))[::-1]
        return (lmbdas * abs_x_sorted).sum()

    def prox_1D(self, y, lmbda, eta=1, gamma=None):
        if y > lmbda * eta:
            return y - lmbda * eta
        elif y < - lmbda * eta:
            return y + lmbda * eta
        else:
            return 0.


class SortedMCPPenalty(SortedPenaltyBase):
    def __init__(self):
        self.proxOpe = SortedProx(name_of_prox="smcp")
        name = "Sorted MCP"

    def penalty(self, x, lmbdas, gamma=None):
        abs_x_sorted = np.sort(np.abs(x))[::-1]
        set_indices = abs_x_sorted < gamma * lmbdas
        value = np.zeros_like(x)
        value[set_indices] = (lmbdas * (abs_x_sorted - abs_x_sorted **
                                        2 / (2 * gamma)))[set_indices]
        value[~set_indices] = (lmbdas**2 * gamma / 2.)[~set_indices]
        return value.sum()

    def prox_1D(self, y, lmbda, eta=1, gamma=None):
        if np.abs(y) <= lmbda * eta:
            return 0.
        if np.abs(y) > lmbda * gamma:
            return y
        return np.sign(y) * (np.abs(y) - lmbda * eta) / (1. - eta/gamma)


class SortedL05Penalty(SortedPenaltyBase):
    def __init__(self):
        self.proxOpe = SortedProx(name_of_prox="sortedL05")
        name = "Sorted L05"

    def penalty(self, x, lmbdas, gamma=None):
        return (lmbdas * np.sqrt(np.sort(np.abs(x))[::-1])).sum()

    def prox_1D(self, y, lmbda, eta=1, gamma=None):
        t = (3./2.) * (eta * lmbda) ** (2./3.)
        if np.abs(y) < t:
            return 0.
        return y * (2./3.) * (1 + np.cos((2./3.) * np.arccos(
            -(3.**(3./2.)/4.) * (lmbda*eta) * np.abs(y)**(-3./2.))))


class SortedSCADPenalty(SortedPenaltyBase):
    """
    Variable Selection via Nonconcave Penalized Likelihood and Its Oracle Properties
    Fan & Li
    https://www.jstor.org/stable/3085904
    """

    def __init__(self):
        name = "Sorted SCAD"

    def penalty(self, x, lmbdas, gamma=None):
        # from skglm
        abs_x_sorted = np.sort(np.abs(x))[::-1]
        value = np.full_like(x, lmbdas ** 2 * (gamma + 1) / 2)
        for j in range(len(x)):
            if abs_x_sorted[j] <= lmbdas[j]:
                value[j] = lmbdas[j] * abs_x_sorted[j]
            elif abs_x_sorted[j] <= lmbdas[j] * gamma:
                value[j] = (
                    2 * gamma * lmbdas[j] * abs_x_sorted[j]
                    - x[j] ** 2 - lmbdas[j] ** 2) / (2 * (gamma - 1))
        return np.sum(value)

    def prox_1D(self, y, lmbda, eta=1, gamma=None):
        abs_y = np.abs(y)
        if abs_y <= 2 * lmbda:
            return np.sign(y) * np.maximum(0, abs_y - lmbda)
        elif 2 * lmbda < abs_y <= gamma * lmbda:
            return 1/(gamma-2) * ((gamma-1) * y - np.sign(y) * gamma * lmbda)
        return y


class SortedLogSumPenalty(SortedPenaltyBase):
    """
    The Proximity Operator of The Log-sum Penalty,
    Prater-Benette and al.
    https://arxiv.org/pdf/2103.02681.pdf
    """

    def __init__(self):
        self.proxOpe = SortedProx(name_of_prox="logsum")
        name = "Sorted Log-sum"

    def penalty(self, x, lmbdas, gamma=None):
        abs_x_sorted = np.sort(np.abs(x))[::-1]
        return (lmbdas * np.log(1 + abs_x_sorted / gamma)).sum()

    def prox_1D(self, y, lmbda, eta=1, gamma=None):
        # TODO: gérer eta

        abs_y = np.abs(y)
        if np.sqrt(lmbda) <= gamma:
            # Prox-convex case
            t = lmbda / gamma
            if abs_y <= t:
                return 0.
            else:
                return np.sign(y) * (0.5 * (abs_y - gamma) +
                                     np.sqrt(0.25*(abs_y + gamma)**2 - lmbda))

        else:
            # Non-convex case
            def root_to_compute(t):
                r2 = 0.5 * (t - gamma) + np.sqrt(0.25*(t + gamma)**2 - lmbda)
                fun = (0.5 / lmbda) * ((r2 - t)**2 - t ** 2) + \
                    np.log(1 + r2 / gamma)
                return fun
            low_bound = 2 * np.sqrt(lmbda) - gamma
            up_bound = lmbda / gamma
            thresh = bisect(root_to_compute, low_bound, up_bound)
            if abs_y <= thresh:
                return 0.
            else:
                return np.sign(y) * (0.5 * (abs_y - gamma) +
                                     np.sqrt(0.25*(abs_y + gamma)**2 - lmbda))
