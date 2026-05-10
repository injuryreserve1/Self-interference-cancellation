import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
# gdown.download(url, downloaded_file, quiet=False, fuzzy=True)
# it does not work w/ fuzzy=True
gdown.download(url, downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def spatial_cancel(signal, alpha=0.6):
    band = np.zeros_like(signal)
    for ch in range(4):
        band[:, ch] = helpers["score_filter"](signal[:, ch])
    
    power = np.mean(np.abs(band)**2, axis=0)
    inv_power = 1.0 / (power + 1e-6)
    w = inv_power / np.max(inv_power)
    
    weighted_energy = np.sum(np.abs(band * w.reshape(1, -1))**2, axis=1)
    thresh = np.percentile(weighted_energy, 99)
    mask = weighted_energy >= thresh
    if mask.sum() < 8:
        mask = slice(None)
    
    selected = band[mask, :] * w.reshape(1, -1)
    cov = selected.conj().T @ selected
    cov += 1e-4 * np.trace(cov) * np.eye(4)
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    
    s = band @ v
    denom = np.vdot(s, s) + 1e-30
    e_pred = np.zeros_like(signal)
    for ch in range(4):
        scale = np.vdot(s, band[:, ch]) / denom
        e_pred[:, ch] = scale * s
    
    return signal - alpha * e_pred


def your_canceller(tx_n, rx):
    tmp = rx - helpers["fit_tx_prediction"](rx)
    tmp = spatial_cancel(tmp)
    tmp = tmp - helpers["fit_tx_prediction"](tmp)
    res = spatial_cancel(tmp)
    return res


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
