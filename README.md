# Reproducing experiments for "Error Estimation of Spectral Clustering in the Stochastic Block Model"

## Requirements

```
numpy matplotlib scikit-learn
```

## Files

- `utils.py` SBM graph generation, spectral clustering, and Bayesian GMM fitting
- `plots.py` sweeps over cluster size ratio and p-q gap; scatter plots of predicted vs. empirical error
- `ece_vs_gap.py` calibration experiment: ECE of Bayesian GMM confidence vs. spectral gap

## Running

```bash
python plots.py       # generates error rate figures
python ece_vs_gap.py  # generates ECE vs. gap figure
```
