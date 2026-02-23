import numpy as np
from itertools import permutations
from sklearn.mixture import BayesianGaussianMixture


def sbm_kclass(n, fracs, p, q, seed=None):
    rng = np.random.default_rng(seed)
    k = len(fracs)
    labels = np.zeros(n, dtype=np.int64)
    boundaries = np.concatenate([[0], np.round(np.cumsum(fracs) * n).astype(int)])
    boundaries[-1] = n
    for c in range(k):
        labels[boundaries[c]:boundaries[c + 1]] = c
    rng.shuffle(labels)
    same = labels[:, None] == labels[None, :]
    P = np.where(same, p, q).astype(float)
    np.fill_diagonal(P, 0.0)
    A = (rng.random((n, n)) < P).astype(np.int8)
    A = np.triu(A, 1)
    A = A + A.T
    return A, labels


def calc_error_rate(true_labels, preds):
    k = int(max(true_labels.max(), preds.max())) + 1
    best_err = 1.0
    for perm in permutations(range(k)):
        mapped = np.array([perm[p] for p in preds])
        err = (mapped != true_labels).mean()
        best_err = min(best_err, err)
    return best_err


def fit_bayes_gmm(X, k):
    d = X.shape[1]
    gmm = BayesianGaussianMixture(n_components=k,
                                  covariance_type='full',
                                  max_iter=5000,
                                  weight_concentration_prior_type='dirichlet_distribution',
                                  weight_concentration_prior=0.05,
                                  mean_precision_prior=3.0,
                                  degrees_of_freedom_prior=d + 10)
    gmm.fit(X)
    return gmm


def pred_error_from_gmm_mc(gmm, n_samples=100000):
    samples, labels = gmm.sample(n_samples)
    preds = gmm.predict(samples)
    return (preds != labels).mean()


def spect_clustering(A, k):
    _, eigvecs = np.linalg.eigh(A)
    if k == 2:
        X = eigvecs[:, -2].reshape(-1, 1)
    else:
        X = eigvecs[:, -1:-k - 1:-1]
    gmm = fit_bayes_gmm(X, k)
    preds = gmm.predict(X)
    pred_err = pred_error_from_gmm_mc(gmm)
    return preds, pred_err
