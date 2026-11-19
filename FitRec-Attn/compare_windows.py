"""Fair comparison of T = 10 and T = 20 on the same test windows.

    python compare_windows.py

The T = 20 run is tested on fewer windows than the T = 10 one: its first prediction in
each workout is step 20, not step 10, so the first (hardest) steps of every workout are
left out. Here every model is evaluated on exactly the T = 20 test windows:

- Mamba T = 20 sees the whole 20-step window;
- Mamba T = 10 sees only its last 10 steps (the same target, 9 previous heart rates);
- trivial (repeat last HR), linear with 9 lags (baselines.json) and with 19 lags
  (baselines_T20.json, run `python baselines.py --T 20` first).

Writes figures/cache/compare_T10_T20.json.
"""

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
STATES = "data/fitrec-attn/model_states/"
RUN10 = STATES + "mamba_userId_context_lrdecay0.7_2026-11-17-20-36-36"
RUN20 = STATES + "mamba_T20_userId_context_lrdecay0.7_2026-11-18-11-47-39"
OUT = "figures/cache/compare_T10_T20.json"


def load_module():
    spec = importlib.util.spec_from_file_location("fitrec_attn", "fitrect-attn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(fa, run_dir):
    import torch

    with open(run_dir + "/config.json") as f:
        cfg = json.load(f)
    keys = ("include_user", "include_sport", "include_temporal", "lr_decay", "arch", "mamba_layers", "mamba_d_state")
    m = fa.da_rnn(
        **{k: cfg[k] for k in keys if k in cfg},
        logger=fa.logger, parallel=False, T=cfg.get("T", 10),
        encoder_hidden_size=64, decoder_hidden_size=64, learning_rate=0.005,
        batch_size=12800, resume_dir=run_dir,
    )
    last = torch.load(run_dir + "/last.pt", map_location="cpu", weights_only=False)
    ck = torch.load(last["best_epoch_path"], map_location="cpu", weights_only=False)
    m.encoder.load_state_dict(ck["en"])
    m.context_encoder.load_state_dict(ck["context_en"])
    m.decoder.load_state_dict(ck["de"])
    for net in (m.encoder, m.context_encoder, m.decoder):
        net.eval()
    print("{}: best epoch {}".format(os.path.basename(run_dir), last["best_epoch"]))
    return m


def predict(m, batch):
    import torch

    with torch.no_grad():
        attr, c1, c2, x, y_hist, y_tgt = m.get_batch(batch)
        return m.forward(attr, c1, c2, x, y_hist).cpu().numpy().astype(np.float64)


def last_steps(batch, k):
    """The last k steps of every window: a T = k window with the same target."""
    inputs, outputs = batch
    sliced = {key: (v[:, -k:] if key != "workoutid" else v) for key, v in inputs.items()}
    return sliced, outputs[:, -k:]


def main():
    from tqdm import tqdm

    with open("figures/cache/baselines.json") as f:
        b9 = json.load(f)
    with open("figures/cache/baselines_T20.json") as f:
        b19 = json.load(f)
    w9, i9 = np.array(b9["weights"]), b9["intercept"]
    w19, i19 = np.array(b19["weights"]), b19["intercept"]

    fa = load_module()
    m20 = load_model(fa, RUN20)
    m10 = load_model(fa, RUN10)

    names = ("mamba_T20", "mamba_T10", "trivial", "linear_9", "linear_19")
    se = dict.fromkeys(names, 0.0)
    ae = dict.fromkeys(names, 0.0)
    n = 0
    gen = m20.endo_reader.generator_for_autotrain(m20.batch_size, m20.num_steps, "test")
    for batch in tqdm(gen, total=m20.n_batches("test"), desc="test", file=sys.stdout, mininterval=30):
        hr = batch[1][:, :, 0].astype(np.float64)  # batch * 20
        y = hr[:, -1]
        preds = {
            "mamba_T20": predict(m20, batch),
            "mamba_T10": predict(m10, last_steps(batch, 10)),
            "trivial": hr[:, -2],
            "linear_9": hr[:, -10:-1] @ w9 + i9,
            "linear_19": hr[:, :-1] @ w19 + i19,
        }
        for k, p in preds.items():
            e = p - y
            se[k] += float((e**2).sum())
            ae[k] += float(np.abs(e).sum())
        n += len(y)

    res = {"n_windows": n, "first_predicted_step": 20}
    for k in names:
        res[k] = dict(mse=se[k] / n, rmse=(se[k] / n) ** 0.5, mae=ae[k] / n)
        print("{:10s} MSE {:.4f}  RMSE {:.3f}  MAE {:.3f}".format(k, res[k]["mse"], res[k]["rmse"], res[k]["mae"]))
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
