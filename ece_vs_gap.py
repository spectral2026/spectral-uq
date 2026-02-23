import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

from utils import sbm_kclass, fit_bayes_gmm


def align_labels(z_true, z_pred, k):
    best_perm, best_err = None, 1.0
    for perm in permutations(range(k)):
        mapped = np.array([perm[p] for p in z_pred])
        err = (mapped != z_true).mean()
        if err < best_err:
            best_err = err
            best_perm = perm
    return np.array([best_perm[p] for p in z_pred])


def compute_ece(conf, correct, n_bins=20):
    edges = np.linspace(0.6, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    total = len(conf)
    ece = 0.0
    for i in range(len(edges) - 1):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() / total * abs(correct[m].mean() - conf[m].mean())
    return ece


def main():
    n = 1000
    k = 2
    n_trials = 1000
    seed = 42

    pq_settings = [
        (0.30, 0.26),
        (0.30, 0.255),
        (0.30, 0.25),
        (0.30, 0.24),
        (0.30, 0.23),
        (0.30, 0.22),
        (0.30, 0.21),
        (0.30, 0.20),
    ]
    n1_ratios = [0.3, 0.4, 0.5]
    colors = ["blue", "orange", "green"]

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for ratio, color in zip(n1_ratios, colors):
        fracs = [ratio, 1.0 - ratio]
        gaps, ece_means, ece_stds = [], [], []

        for p, q in pq_settings:
            print(f"ratio={ratio}, p={p}, q={q}")
            per_trial = []
            for _ in range(n_trials):
                trial_seed = int(rng.integers(0, 2**31))
                A, z_true = sbm_kclass(n=n, fracs=fracs, p=p, q=q, seed=trial_seed)
                _, eigvecs = np.linalg.eigh(A)
                X = eigvecs[:, -1:-k - 1:-1]
                gmm = fit_bayes_gmm(X, k)
                probs = gmm.predict_proba(X)
                z_pred = probs.argmax(axis=1)
                conf = probs.max(axis=1)
                z_aligned = align_labels(z_true, z_pred, k)
                correct = (z_aligned == z_true).astype(int)
                per_trial.append(compute_ece(conf, correct))

            gaps.append(p - q)
            ece_means.append(np.mean(per_trial))
            ece_stds.append(np.std(per_trial))

        gaps = np.array(gaps)
        ece_means = np.array(ece_means)
        ece_stds = np.array(ece_stds)

        ax.errorbar(gaps, ece_means, yerr=ece_stds,
                    fmt="o-", color=color,
                    markersize=7, linewidth=2,
                    capsize=4, capthick=1.5,
                    label=f"$n_1/n={ratio}$")

    ax.set_xlabel("$p - q$")
    ax.set_ylabel("ECE")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
