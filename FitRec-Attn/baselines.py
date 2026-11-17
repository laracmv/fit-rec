"""Simple heart-rate baselines on the same windows the model sees (T=10: 9 past HR -> next HR).

    python baselines.py

- trivial:  y_hat = HR[t+8]           (repeat the last observed heart rate)
- linear:   y_hat = w . HR[t..t+8] + b  (OLS fitted on all train windows)

Windows are built exactly like dataInterpreter.generator_for_autotrain: every start t in
0..len-T-1 of each workout, in split order, keeping only the first floor(N/batch)*batch windows.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = 10
BATCH = 12800  # same batch size as training/evaluation, so the window set matches the model's
CHUNK = 2000  # workouts per chunk


def hr_windows(c, hr_ch, rows):
    """Yield (X, y) chunks: X = 9 past HR values, y = HR at the next step."""
    lengths = c["lengths"]
    for i in range(0, len(rows), CHUNK):
        rr = rows[i : i + CHUNK]
        Xs, ys = [], []
        for r in rr:
            n = int(lengths[r]) - T
            if n <= 0:
                continue
            hr = c["seq"][r, : n + T - 1, hr_ch].astype(np.float64)
            w = np.lib.stride_tricks.sliding_window_view(hr, T)  # n * T
            Xs.append(w[:, : T - 1])
            ys.append(w[:, T - 1])
        if Xs:
            yield np.concatenate(Xs), np.concatenate(ys)


def windows_limited(c, hr_ch, rows):
    """Same as hr_windows but truncated to whole batches, as the generator does."""
    total = int(np.maximum(c["lengths"][rows] - T, 0).sum())
    keep = (total // BATCH) * BATCH
    seen = 0
    for X, y in hr_windows(c, hr_ch, rows):
        if seen >= keep:
            break
        take = min(len(y), keep - seen)
        seen += take
        yield X[:take], y[:take]


def main():
    path = os.path.join(HERE, "data", "fitrec_compact.npz")
    with np.load(path, allow_pickle=False) as f:
        c = {k: f[k] for k in ("seq", "lengths", "channels", "train_rows", "valid_rows", "test_rows")}
    hr_ch = [str(x) for x in c["channels"]].index("tar_heart_rate")

    # OLS via normal equations: accumulate [X 1]^T [X 1] and [X 1]^T y over all train windows
    d = T  # 9 lags + intercept
    A = np.zeros((d, d))
    b = np.zeros(d)
    n_train = 0
    for X, y in windows_limited(c, hr_ch, c["train_rows"]):
        Xb = np.hstack([X, np.ones((len(X), 1))])
        A += Xb.T @ Xb
        b += Xb.T @ y
        n_train += len(y)
    coef = np.linalg.solve(A, b)
    print("train windows: {}".format(n_train))
    print("weights (HR[t-9] .. HR[t-1]): {}".format(np.round(coef[:-1], 4).tolist()))
    print("intercept: {:.4f}   sum of weights: {:.4f}".format(coef[-1], coef[:-1].sum()))

    results = {"weights": coef[:-1].tolist(), "intercept": float(coef[-1]), "n_train_windows": n_train}
    for split in ("train", "valid", "test"):
        n, se_triv, ae_triv, se_lin, ae_lin = 0, 0.0, 0.0, 0.0, 0.0
        for X, y in windows_limited(c, hr_ch, c[split + "_rows"]):
            e_triv = X[:, -1] - y
            e_lin = X @ coef[:-1] + coef[-1] - y
            n += len(y)
            se_triv += (e_triv**2).sum()
            ae_triv += np.abs(e_triv).sum()
            se_lin += (e_lin**2).sum()
            ae_lin += np.abs(e_lin).sum()
        r = dict(
            n_windows=n,
            trivial_mse=se_triv / n, trivial_rmse=(se_triv / n) ** 0.5, trivial_mae=ae_triv / n,
            linear_mse=se_lin / n, linear_rmse=(se_lin / n) ** 0.5, linear_mae=ae_lin / n,
        )
        results[split] = r
        print("{:5s} n={:8d}  trivial MSE {:.4f} (RMSE {:.3f})   linear MSE {:.4f} (RMSE {:.3f})".format(
            split, n, r["trivial_mse"], r["trivial_rmse"], r["linear_mse"], r["linear_rmse"]))

    os.makedirs(os.path.join(HERE, "figures", "cache"), exist_ok=True)
    with open(os.path.join(HERE, "figures", "cache", "baselines.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
