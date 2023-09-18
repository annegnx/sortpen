import numpy as np
from time import time

from sklearn.isotonic import isotonic_regression
from scipy.optimize import root, bisect


class IsoProxBase:

    def __call__(self,  v: np.ndarray, lmbdas: np.ndarray, eta, gamma) -> np.ndarray:
        # init
        x = self.prox_init(v, lmbdas, eta, gamma)
        # print("Init (unconstrained)", x)
        b_s = 1
        b_ = 0
        N = len(x)
        while b_s < N:
            if x[b_] >= x[b_s]:
                b_, b_s = b_s, b_s+1
            else:

                if b_s + 1 < N:
                    x_prev = x[b_s]
                    b_s = b_s+1

                    while x_prev <= x[b_s]:

                        if b_s + 1 == N:
                            break
                        else:
                            x_prev = x[b_s]
                            b_s = b_s+1

                    b_s -= 1

                x_prev = x.copy()

                x = self.merge(x, eta, lmbdas, v, gamma, b_, b_s)
                # print("After merge", x)
                if b_ > 0:
                    b_l = b_ - 1

                    while (x[b_l] < x[b_s]):

                        x = self.merge(x, eta, lmbdas, v, gamma, b_l, b_s)

                        if b_l > 0:
                            b_l -= 1
                        else:
                            break

                b_, b_s = b_s, b_s+1
        return x


class IsoProxSlope(IsoProxBase):
    def prox_init(self, v, lmbda, eta, gamma):
        return v - eta * lmbda

    def merge(self, x, eta, lmbda, v, gamma, b_, b_sup):
        mean_value = x[b_:b_sup+1].mean()
        x[b_:b_sup+1] = mean_value
        return x


class IsoProxLasso():
    def __call__(self, v, lmbda):
        return np.sign(v) * np.maximum(np.abs(v) - lmbda, 0)


class IsoProxLCA(IsoProxBase):

    def prox_init(self, v, lmbda, eta, gamma):
        return np.minimum(v-eta * lmbda, v / (1+eta/gamma))

    def merge(self, x, eta, lmbda, v, gamma, b_, b_sup):
        l = np.array([lmbda[k]*((b_sup - b_ + 1) * gamma + eta * (b_sup - k)) +
                     eta * lmbda[b_:k+1].sum() for k in range(b_, b_sup+1)])

        diff = ((l - v[b_:b_sup+1].sum()) <= 0).astype(int)

        max_ = max(diff)
        min_ = min(diff)

        s_star = (b_ - 1 + np.argmax(diff)) * max_ * \
            (1-min_) + (1-max_) * b_sup + min_ * (b_-1)

        mean_value = (v[b_:b_sup+1].sum() - eta * lmbda[b_: s_star+1].sum()) / \
            ((b_sup - b_ + 1) + eta / gamma * ((b_sup - s_star)))

        x[b_:b_sup+1] = mean_value

        return x


class IsoProxLCAImplicit(IsoProxBase):
    def prox_init(self, v, lmbda, eta, gamma):
        return np.minimum(v-eta * lmbda, v / (1+eta/gamma))

    def prox_implicit(self, t,  eta, lmbda, v, gamma, b_, b_sup):
        f = v[b_:b_sup+1].sum() - eta * (lmbda[b_:b_sup+1] *
                                         (t < gamma * lmbda[b_:b_sup+1])).sum()
        f /= (b_sup-b_+1) + eta / gamma * \
            (t >= gamma * lmbda[b_:b_sup+1]).astype(int).sum()
        return f-t

    def merge(self, x, eta, lmbda, v, gamma, b_, b_sup):

        mean_value = root(lambda t: self.prox_implicit(
            t,  eta, lmbda, v, gamma, b_, b_sup), x[b_])
        x[b_:b_sup+1] = mean_value.x

        return x


class IsoProxSMCP(IsoProxBase):
    def __init__(self) -> None:
        def initialization(v, lmbda, eta, gamma):
            return np.minimum(v, (v - eta * lmbda) / (1-eta/gamma))
        self.prox_init = initialization

    def merge(self, x, eta, lmbda, v, gamma, b_, b_sup):
        l = np.array([lmbda[k]*((b_sup - b_ + 1) * gamma + eta * (b_ - k)) +
                     eta * lmbda[b_:k].sum() for k in range(b_, b_sup+1)])
        diff = ((l - v[b_:b_sup+1].sum()) <= 0).astype(int)
        max_ = max(diff)
        min_ = min(diff)
        s_star = (b_ - 1 + np.argmax(diff)) * max_ * \
            (1-min_) + (1-max_) * b_sup + min_ * (b_-1)
        mean_value = (v[b_:b_sup+1].sum() - eta * lmbda[b_: s_star].sum()) / \
            ((b_sup - b_ + 1) + eta / gamma * ((b_ - s_star + 1)))
        x[b_:b_sup+1] = mean_value
        return x


class IsoProxSMCPImplicit(IsoProxBase):
    def __init__(self) -> None:
        def initialization(v, lmbda, eta, gamma):
            return np.minimum(v, (v - eta * lmbda) / (1-eta/gamma))
        self.prox_init = initialization

    def proxSMCP_implicit(self, t,  eta, lmbda, v, gamma, b_, b_sup):
        f = (b_sup - b_ + 1) * t - v[b_:b_sup+1].sum() + eta * \
            (np.maximum(lmbda[b_:b_sup+1] - t/gamma,
             np.zeros_like(lmbda[b_:b_sup+1]))).sum()
        return f

    def merge(self, x, eta, lmbda, v, gamma, b_, b_sup):
        mean_value = root(lambda t: self.proxSMCP_implicit(
            t,  eta, lmbda, v, gamma, b_, b_sup), x[b_])
        x[b_:b_sup+1] = mean_value.x
        return x


class IsoProxL_05(IsoProxBase):

    def objective(self, x, y, lmbdas, eta):
        return (1./(2. * eta)) * np.linalg.norm(x-y)**2 + (lmbdas * np.sqrt(np.abs(x))).sum()

    def update(self, x,  lmbda):
        t = (3./4.**(2./3)) * (lmbda) ** (2./3.)

        if np.abs(x) <= t:
            return 0.

        return x * (2./3.) * (1 + np.cos((2./3.) * np.arccos(
            -(3.**(3./2.)/4.) * (lmbda) * np.abs(x)**(-3./2.))))

    def mus(self, x, y, lmbdas):
        mus = np.zeros(len(x))
        mus[0] = x[0] - y[0] + lmbdas[0]/(2*np.sqrt(x[0]))
        for i in range(1,  len(x)):
            mus[i] = mus[i-1] + x[i] - y[i] + \
                lmbdas[i] / (2*np.sqrt(np.abs(x[i])))
        return mus

    def prox_init(self, v, lmbda, eta, gamma):
        x = np.zeros_like(v)
        for i, (y_, lmbda_) in enumerate(zip(v, lmbda * np.ones_like(v) * eta)):
            x[i] = self.update(y_, lmbda_ * eta)
        return x

    def merge(self, x, eta, lmbda, v, gamma, b_, b_sup):
        mean_value = self.update(
            v[b_:b_sup+1].mean(), eta * lmbda[b_:b_sup+1].mean())
        x[b_:b_sup+1] = mean_value
        return x

    def __call__(self, v, lmbdas, eta, gamma, verbose=False):
        n = len(v)
        res = super().__call__(v, lmbdas, eta, gamma)
        tmp_objective_prev = self.objective(res, v, lmbdas, eta)
        if verbose:
            print()
            print("Full res", res)
            print("Full res obj", tmp_objective_prev)
            print("Full res mu", self.mus(res, v, lmbdas))
        for index_zeros in range(n):

            tmp_res = np.zeros(n)
            tmp_res[:-index_zeros] = super().__call__(v[:-index_zeros],
                                                      lmbdas[:-index_zeros], eta, gamma)

            tmp_objective = self.objective(tmp_res, v, lmbdas, eta)
            if verbose:
                print()
                print("Index", index_zeros)
                print("Full res", tmp_res)
                print("Full res obj", tmp_objective)
                print("Full res mu", self.mus(tmp_res, v, lmbdas))
            if tmp_objective < tmp_objective_prev:

                res = tmp_res.copy()
                tmp_objective_prev = tmp_objective
        return res


class IsoProxLogSum(IsoProxBase):
    def objective(self, x, y, lmbdas, eta, gamma):
        return (1./(2. * eta)) * np.linalg.norm(x-y)**2 + (lmbdas * np.log(1+abs(x)/gamma)).sum()

    def update(self, x,  lmbda, gamma):
        abs_x = np.abs(x)
        if np.sqrt(lmbda) <= gamma:
            # Prox-convex case

            t = lmbda / gamma
            if abs_x <= t:
                return 0.
            else:
                return np.sign(x) * (0.5 * (abs_x - gamma) +
                                     np.sqrt(0.25*(abs_x + gamma)**2 - lmbda))

        else:
            # Non-convex case
            # TODO: check si c'est le bon threshold
            thresh = 2 * np.sqrt(lmbda) - gamma
            if abs_x <= thresh:
                return 0.
            else:

                return np.sign(x) * (0.5 * (abs_x - gamma) +
                                     np.sqrt(0.25*(abs_x + gamma)**2 - lmbda))

    def prox_init(self, v, lmbda, eta, gamma):
        x = np.zeros_like(v)
        for i, (y_, lmbda_) in enumerate(zip(v, lmbda * np.ones_like(v) * eta)):
            x[i] = self.update(y_, lmbda_ * eta, gamma)
        return x

    def merge(self, x, eta, lmbda, v, gamma, b_, b_sup):
        mean_value = self.update(
            v[b_:b_sup+1].mean(), eta * lmbda[b_:b_sup+1].mean(), gamma)
        x[b_:b_sup+1] = mean_value
        return x

    def __call__(self, v, lmbdas, eta, gamma, verbose=False):
        n = len(v)
        res = super().__call__(v, lmbdas, eta, gamma)
        tmp_objective_prev = self.objective(res, v, lmbdas, eta, gamma)
        if verbose:
            print()
            print("Full res", res)
            print("Full res obj", tmp_objective_prev)
            print("Full res mu", self.mus(res, v, lmbdas))
        for index_zeros in range(n):

            tmp_res = np.zeros(n)
            tmp_res[:-index_zeros] = super().__call__(v[:-index_zeros],
                                                      lmbdas[:-index_zeros], eta, gamma)

            tmp_objective = self.objective(tmp_res, v, lmbdas, eta, gamma)
            if verbose:
                print()
                print("Index", index_zeros)
                print("Full res", tmp_res)
                print("Full res obj", tmp_objective)
                print("Full res mu", self.mus(tmp_res, v, lmbdas))
            if tmp_objective < tmp_objective_prev:

                res = tmp_res.copy()
                tmp_objective_prev = tmp_objective
        return res


dict_prox = {"slope": IsoProxSlope(),
             "smcp": IsoProxSMCPImplicit(),
             "lca_smcp": IsoProxLCA(),
             "sortedL05":  IsoProxL_05(),
             "logsum": IsoProxLogSum(),
             }


class SortedProx():

    def __init__(self, name_of_prox):
        self.isoProx = dict_prox[name_of_prox]

    def __call__(self, v: np.ndarray, lmbdas: np.ndarray, eta=1, gamma=None):
        abs_v = np.abs(v)
        indices = np.argsort(abs_v)[::-1]
        P = np.eye(len(v))[:, indices]
        tmp_prox = self.isoProx(
            v=abs_v[indices], lmbdas=lmbdas, eta=eta, gamma=gamma)
        prox = np.sign(v) * np.clip(P @ tmp_prox, 0, None)
        return prox
