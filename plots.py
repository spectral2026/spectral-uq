import numpy as np
import matplotlib.pyplot as plt

from utils import sbm_kclass, calc_error_rate, spect_clustering


def sweep_n1(n, n1_list, p, q, M=20, seed=None):
    mean_err, std_err = [], []
    mean_pred_err, std_pred_err = [], []

    for n1 in n1_list:
        errs, pred_errs = [], []
        for _ in range(M):
            A, true_labels = sbm_kclass(n=n, fracs=[n1 / n, 1 - n1 / n], p=p, q=q)
            preds, pred_err = spect_clustering(A=A, k=2)
            errs.append(calc_error_rate(true_labels=true_labels, preds=preds))
            pred_errs.append(pred_err)

        mean_err.append(float(np.mean(errs)))
        std_err.append(float(np.std(errs) if M > 1 else 0.0))
        mean_pred_err.append(float(np.mean(pred_errs)))
        std_pred_err.append(float(np.std(pred_errs) if M > 1 else 0.0))

    mean_err = np.array(mean_err) * 100.0
    std_err = np.array(std_err) * 100.0
    mean_pred_err = np.array(mean_pred_err) * 100.0
    std_pred_err = np.array(std_pred_err) * 100.0
    n1_ratio = np.asarray(n1_list, dtype=float) / float(n)

    colors = ['skyblue', 'salmon']
    plt.figure(figsize=(7, 5))
    plt.plot(n1_ratio, mean_err, '--o', label="empirical", color=colors[0], markersize=5)
    plt.fill_between(n1_ratio, mean_err - std_err, mean_err + std_err, alpha=0.1, color=colors[0])
    plt.plot(n1_ratio, mean_pred_err, '--s', label="predicted", color=colors[1], markersize=5)
    plt.fill_between(n1_ratio, mean_pred_err - std_pred_err, mean_pred_err + std_pred_err, alpha=0.1, color=colors[1])
    plt.xlabel(r"$n_1/n$")
    plt.ylabel("error rate [%]")
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()


def sweep_pq_gap(n, n1, p_list, q, M=20, seed=None):
    rng = np.random.default_rng(seed)
    mean_err, std_err = [], []
    mean_pred_err, std_pred_err = [], []

    for p in p_list:
        errs, pred_errs = [], []
        for _ in range(M):
            trial_seed = int(rng.integers(0, 2**32 - 1)) if seed is not None else None
            A, true_labels = sbm_kclass(n=n, fracs=[n1 / n, 1 - n1 / n],
                                        p=float(p), q=float(q), seed=trial_seed)
            preds, pred_err = spect_clustering(A=A, k=2)
            errs.append(calc_error_rate(true_labels=true_labels, preds=preds))
            pred_errs.append(pred_err)

        mean_err.append(float(np.mean(errs)))
        std_err.append(float(np.std(errs) if M > 1 else 0.0))
        mean_pred_err.append(float(np.mean(pred_errs)))
        std_pred_err.append(float(np.std(pred_errs) if M > 1 else 0.0))

    mean_err = np.array(mean_err) * 100.0
    std_err = np.array(std_err) * 100.0
    mean_pred_err = np.array(mean_pred_err) * 100.0
    std_pred_err = np.array(std_pred_err) * 100.0
    pq_gap = np.asarray(p_list, dtype=float) - float(q)

    colors = ['skyblue', 'salmon']
    plt.figure(figsize=(7, 5))
    plt.plot(pq_gap, mean_err, '--o', label="empirical", color=colors[0], markersize=5)
    plt.fill_between(pq_gap, mean_err - std_err, mean_err + std_err, alpha=0.1, color=colors[0])
    plt.plot(pq_gap, mean_pred_err, '--s', label="predicted", color=colors[1], markersize=5)
    plt.fill_between(pq_gap, mean_pred_err - std_pred_err, mean_pred_err + std_pred_err, alpha=0.1, color=colors[1])
    plt.xlabel(r"$p-q$")
    plt.ylabel("error rate [%]")
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()

    return pq_gap, mean_err, std_err, mean_pred_err, std_pred_err


def scatter_pred_vs_err(n, k, pmin, pmax, q, M=200, seed=None):
    rng = np.random.default_rng(seed)
    errs, pred_errs = [], []
    min_frac = 0.25 if k >= 3 else 0.20

    for ii in range(M):
        if ii % 50 == 0:
            print(f"  sample {ii}/{M}")
        raw = rng.dirichlet(np.ones(k))
        fracs = raw * (1.0 - k * min_frac) + min_frac
        p = float(rng.uniform(pmin, pmax))
        A, true_labels = sbm_kclass(n=n, fracs=fracs, p=p, q=q)
        preds, pred_err = spect_clustering(A=A, k=k)
        errs.append(calc_error_rate(true_labels=true_labels, preds=preds))
        pred_errs.append(pred_err)

    errs = np.asarray(errs) * 100.0
    pred_errs = np.asarray(pred_errs) * 100.0

    lo = float(min(errs.min(), pred_errs.min()))
    hi = float(max(errs.max(), pred_errs.max()))
    pad = 0.03 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad

    plt.figure(figsize=(6.5, 5))
    plt.scatter(errs, pred_errs, s=35, alpha=0.8, color='salmon', edgecolor="none", label="samples")
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2.0, color="k", alpha=0.7, label=r"$y=x$")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("empirical error [%]")
    plt.ylabel("predicted error [%]")
    plt.title(f"{k}-cluster SBM")
    plt.grid()
    plt.legend()
    plt.tight_layout()


def main():
    n = 1000
    p = 0.38
    q = 0.3
    M = 20
    seed = 123

    n1_list = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.65]) * n
    sweep_n1(n=n, n1_list=n1_list, p=p, q=q, M=M, seed=seed)

    n1 = 0.5 * n
    p_list = np.array([0.3 + g for g in [0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05, 0.045, 0.04]])
    sweep_pq_gap(n=n, n1=n1, p_list=p_list, q=q, M=M, seed=seed)

    for k in [2, 3, 4]:
        print(f"Running scatter k={k}...")
        pmin = 0.4 if k == 4 else 0.37
        scatter_pred_vs_err(n=n, k=k, pmin=pmin, pmax=0.6, q=q, M=200, seed=seed)

    plt.show()


if __name__ == "__main__":
    main()
