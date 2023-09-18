from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sortpen.loss import grad_reg_loss


def find_clusters_1D(y, tol=1e-6):
    dic_of_clusters = {}
    old_y = None
    current_i = 0
    for i, elmnt in enumerate(y):
        if i > 0:
            if np.abs(old_y-elmnt) < tol:
                dic_of_clusters[current_i]["end"] += 1
            else:
                current_i += 1
                dic_of_clusters[current_i] = {
                    "start": i, "end": i, "value": elmnt}

        else:
            dic_of_clusters[i] = {"start": i, "end": i, "value": elmnt}
        old_y = elmnt
    return dic_of_clusters


def compute_ratio_clusters(y1, y2):
    true_clusters = find_clusters_1D(y1)
    n_clusters = len(true_clusters)
    clusters_ratio = np.zeros(n_clusters)
    for i in true_clusters:
        dic_cluster = true_clusters[i]
        if dic_cluster["value"] != 0:  # on sépare clusterisation de sparsite
            seq = y2[dic_cluster["start"]:dic_cluster["end"]+1]
            count = Counter(seq)
            clusters_ratio[i] = 1 - (len(count)-1) / \
                max(1, (dic_cluster["end"]-dic_cluster["start"]))
    return clusters_ratio.mean()


def clustered_ratio_1D(pred_y, true_y):
    intra_ratio = compute_ratio_clusters(true_y, pred_y)
    inter_ratio = compute_ratio_clusters(pred_y, true_y)
    # n_equal_values_count = len(Counter(true_y))
    # pred_equal_values_count = len(Counter(pred_y))
    # p = len(true_y)
    # if n_equal_values_count == 1:
    #     inter_clusters_ratio = (p-pred_equal_values_count) / (
    #         p-n_equal_values_count)
    # elif n_equal_values_count == p:
    #     inter_clusters_ratio = (
    #         pred_equal_values_count - 1) / (n_equal_values_count-1)
    # else:
    #     inter_clusters_ratio = min(np.abs(p-pred_equal_values_count) / (
    #         p-n_equal_values_count), np.abs(pred_equal_values_count - 1) / (n_equal_values_count-1))

    rmse = np.sqrt(((pred_y-true_y)**2).sum()) / np.linalg.norm(true_y)
    res = {"intra_clusters_ratio": intra_ratio,
           "inter_clusters_ratio": inter_ratio, "rmse": rmse, "f1_score": f1_score(intra_ratio, inter_ratio)}
    return res


def clustering(y_pred, y_true):

    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    counts_pred = counts_pred[np.where(unique_pred != 0)[0]]
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    counts_true = counts_true[np.where(unique_true != 0)[0]]
    f1_score = len(counts_pred) / len(counts_true)
    return 1 / (f1_score + 1./f1_score)


def sparsity(pred_y, true_y):

    sparse_pred = (pred_y == 0.)
    sparse_true = (true_y == 0.)
    recall = (sparse_pred[sparse_true]).sum() / sparse_true.sum()
    precision = (sparse_pred[sparse_true]).sum() / sparse_pred.sum()
    res = {"sparse_recall": recall,
           "sparse_precision": precision,  "f1_score": f1_score(precision, recall)}
    return res


def rmse(pred_y, true_y):
    return np.sqrt(((pred_y-true_y)**2).sum()) / np.linalg.norm(true_y)


def mse(pred_y, true_y):
    return ((pred_y-true_y)**2).sum()


def pairing_clustering(y_pred, y_true):
    cluster_data_pred = (y_pred[None, :] == y_pred[:, None]).flatten()
    cluster_data_truth = (y_true[None, :] == y_true[:, None]).flatten()
    return recall_score(cluster_data_truth, cluster_data_pred)


def f1_score(precision, recall):
    return 2./(1./precision+1./recall)


def subdiff_distance(X, y, beta, lmbdas, gamma):
    dist = beta / gamma - grad_reg_loss(X,  beta, y)
    indices_sorted = np.argsort(np.abs(beta))[::-1]
    beta_sorted = beta[indices_sorted]
    for i, beta_i in enumerate(beta_sorted):
        j = indices_sorted[i]
        if beta_i == 0:
            dist[j] = dist[j]-np.sign(dist[j])*lmbdas[i]
        elif np.abs(beta_i) <= lmbdas[i] * gamma:
            dist[j] -= lmbdas[i] * np.sign(beta_i)
        else:
            dist[j] -= lmbdas[i] * beta_i / gamma
    return np.abs(dist[j])
