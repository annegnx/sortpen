import numpy as np
import sortpen.penalties
from tqdm import trange


class FistaSortPen:
    def __init__(self, Penalty: sortpen.penalties.SortedPenaltyBase, grad_loss, lmbdas: np.ndarray, gamma=None, max_iters=100, tol=1e-10) -> None:
        self.penalty = Penalty
        self.grad_loss = grad_loss
        self.lmbdas = lmbdas
        self.iterations = []
        # self.subdiff = []
        self.max_iters = max_iters
        self.gamma = gamma
        self.tol = tol

    def minimization_step(self, b_k, previous_u_k, X, y):
        c_k = b_k - (self.grad_loss(X,  b_k, y)) * (1 / (self.alpha))

        # sign_c = np.sign(c_k)
        # s = np.abs(c_k)
        # indexes = np.argsort(s)[::-1]

        computed_prox = self.penalty.proxOpe(
            v=c_k,  eta=(1 / self.alpha), gamma=self.gamma, lmbdas=self.lmbdas)

        # n = len(computed_prox)
        # computed_prox = np.eye(n)[:, indexes] @ computed_prox

        u_k = computed_prox.copy()

        b_next = u_k + self.zeta * (u_k - previous_u_k)
        previous_u_k = u_k.copy()

        return b_next, previous_u_k

    def __call__(self, X, y):

        self.t = 1
        b = np.zeros(X.shape[1])
        u = np.zeros(X.shape[1])
        for step in trange(self.max_iters):

            old_t = self.t
            self.t = 0.5 * (1+np.sqrt(1+4 * self.t**2))
            self.zeta = (old_t - 1) / self.t
            old_u = u.copy()
            b, u = self.minimization_step(b, u, X, y)
            self.iterations.append(u)
            # self.subdiff.append(subdiff_distance(X, y, u, self.lmbdas, self.gamma))
            # TODO: ajouter critère d'arrêt subdiff_distance aux objets Penalties
            # if step % 100 == 0:
            #     plt.stem(np.arange(X.shape[1]), b)
            #     plt.title("step {} , direct SMCP".format(step))
            #     plt.show(block=False)
            stop_crit = np.max(np.abs(old_u - u))

            if stop_crit < self.tol:
                print(
                    f"Fista SortPen: Convergence (wrt tol) reached after {step  } iterations.")
                break

        return u

    def fit(self, X, y):
        self.alpha = np.linalg.norm(X, ord=2)**2 / len(y)
        b = self(X, y)
        return b, {"iterations": self.iterations}
    # , "subdiff": self.fista_lca.subdiff}
