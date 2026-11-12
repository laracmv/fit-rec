"""Figures for the FitRec-Attn report.

    python make_figures.py compute   # slow: evaluates checkpoints, writes figures/cache/
    python make_figures.py plot      # fast: draws figures/*.png from the cache
    python make_figures.py           # both

`compute` is resumable: each finished piece is cached and skipped on the next run.
"""

import importlib.util
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)  # the training code uses paths relative to FitRec-Attn/

RUN = "userId_context_2026-11-11-18-25-06"
RUN_DIR = "data/fitrec-attn/model_states/" + RUN
BEST_EPOCH = 4
CHECKPOINTS = {e: "{}/{}_epoch_{}".format(RUN_DIR, RUN, e) for e in range(5)}
CHECKPOINTS[7] = RUN_DIR + "/last.pt"  # epochs 5-6 did not improve, so no file was kept
LOGGED_VALID = {4: 8.411466558947835, 7: 8.913}  # printed during training, used as a sanity check

FIG = "figures"
CACHE = FIG + "/cache"
T = 10
TRAIN_SUBSET = 5000  # workouts used to measure the train loss of each checkpoint
OBSERVED = 60  # autoregressive example: true heart rate is given up to this step
SEED = 0

# palette (light mode, reference instance of the dataviz skill)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def load_training_module():
    spec = importlib.util.spec_from_file_location("fitrec_attn", "fitrect-attn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(fa):
    return fa.da_rnn(
        logger=fa.logger,
        parallel=False,
        T=T,
        encoder_hidden_size=64,
        decoder_hidden_size=64,
        learning_rate=0.005,
        batch_size=12800,
        resume_dir=RUN_DIR,
    )


def load_weights(fa, m, path):
    import torch

    ck = torch.load(path, map_location=fa.device, weights_only=False)
    m.encoder.load_state_dict(ck["en"])
    m.context_encoder.load_state_dict(ck["context_en"])
    m.decoder.load_state_dict(ck["de"])
    for net in (m.encoder, m.context_encoder, m.decoder):
        net.eval()


def forward(m, batch):
    """Same computation as da_rnn.evaluate, returning predictions instead of the loss."""
    import torch

    with torch.no_grad():
        attr, c1, c2, x, y_hist, y_tgt = m.get_batch(batch)
        emb = m.context_encoder(c1, c2)
        _, enc = m.encoder(attr, emb, x)
        y_pred = m.decoder(enc, y_hist)
    return y_pred.cpu().numpy(), y_hist.cpu().numpy(), y_tgt.cpu().numpy()


def split_errors(m, split):
    """Squared/absolute error sums of the model and of the 'repeat last heart rate' baseline."""
    from tqdm import tqdm

    acc = dict(n=0, se=0.0, ae=0.0, se_last=0.0, ae_last=0.0)
    gen = m.endo_reader.generator_for_autotrain(m.batch_size, m.num_steps, split)
    for batch in tqdm(gen, total=m.n_batches(split), desc=split, file=sys.stdout, mininterval=10):
        y_pred, y_hist, y_tgt = forward(m, batch)
        err, err_last = y_pred - y_tgt, y_hist[:, -1] - y_tgt
        acc["n"] += len(y_tgt)
        acc["se"] += float((err.astype(np.float64) ** 2).sum())
        acc["ae"] += float(np.abs(err).sum())
        acc["se_last"] += float((err_last.astype(np.float64) ** 2).sum())
        acc["ae_last"] += float(np.abs(err_last).sum())
    return acc


def summarize(acc):
    n = acc["n"]
    return dict(
        n_windows=n,
        mse=acc["se"] / n,
        rmse=(acc["se"] / n) ** 0.5,
        mae=acc["ae"] / n,
        baseline_mse=acc["se_last"] / n,
        baseline_rmse=(acc["se_last"] / n) ** 0.5,
        baseline_mae=acc["ae_last"] / n,
    )


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def compute_curves(fa, m):
    path = CACHE + "/curves.json"
    curves = load_json(path, {})
    rd = m.endo_reader
    rng = np.random.default_rng(SEED)
    rd.split_rows["train_subset"] = np.sort(rng.choice(rd.split_rows["train"], TRAIN_SUBSET, replace=False))
    for epoch, ck in sorted(CHECKPOINTS.items()):
        if str(epoch) in curves:
            continue
        print("\n=== checkpoint epoch {} ===".format(epoch))
        load_weights(fa, m, ck)
        curves[str(epoch)] = {
            "train": summarize(split_errors(m, "train_subset")),
            "valid": summarize(split_errors(m, "valid")),
        }
        save_json(curves, path)
        if epoch in LOGGED_VALID:
            print("valid MSE {:.3f} (logged during training: {:.3f})".format(curves[str(epoch)]["valid"]["mse"], LOGGED_VALID[epoch]))


def compute_test(fa, m):
    path = CACHE + "/test_metrics.json"
    if os.path.exists(path):
        return
    print("\n=== test set, best model (epoch {}) ===".format(BEST_EPOCH))
    load_weights(fa, m, CHECKPOINTS[BEST_EPOCH])
    save_json(summarize(split_errors(m, "test")), path)


def workout_windows(rd, r, starts, y):
    """Batch of windows of workout row r starting at `starts`, with heart rate taken from y."""
    c = rd.compact
    n = len(starts)
    tt = starts[:, None] + np.arange(T)
    g = c["seq"][r][tt]
    cr = c["ctx_row"][r]
    gc = c["seq"][cr][tt]
    inputs = {
        "input": g[:, :, rd.in_ch],
        "user_input": np.full((n, T), c["user_idx"][r]),
        "context_input_1": np.concatenate(
            [gc[:, :, rd.in_ch], np.full((n, T, 1), c["since_last"][r], np.float32)], axis=2
        ),
        "context_input_2": gc[:, :, rd.out_ch],
    }
    return inputs, y[tt][:, :, None].astype(np.float32)


def pick_examples(rd, sport_names, k_per_sport=2):
    """Test workouts with clearly varying heart rate, for the two most common sports."""
    c = rd.compact
    rows = rd.split_rows["test"]
    hr = c["seq"][rows][:, :, rd.out_ch[0]]
    spread = hr[:, 10:].std(axis=1)
    rng = np.random.default_rng(SEED)
    picked = []
    for sport in ("run", "bike"):
        s = sport_names.index(sport)
        cand = rows[(c["sport_idx"][rows] == s) & (spread > np.percentile(spread, 60)) & (spread < np.percentile(spread, 95))]
        picked += [int(x) for x in rng.choice(cand, k_per_sport, replace=False)]
    return picked


def compute_examples(fa, m, sport_names):
    path = CACHE + "/examples.npz"
    if os.path.exists(path):
        return
    print("\n=== example predictions ===")
    load_weights(fa, m, CHECKPOINTS[BEST_EPOCH])
    rd = m.endo_reader
    c = rd.compact
    rows = pick_examples(rd, sport_names)
    L = c["seq"].shape[1]
    true, one_step, rollout = [], [], []
    for r in rows:
        y = c["seq"][r][:, rd.out_ch[0]].astype(np.float64)
        # one step ahead: the model always sees the 9 true previous heart rates
        starts = np.arange(0, L - T + 1)
        pred, _, _ = forward(m, workout_windows(rd, r, starts, y))
        os_ = np.full(L, np.nan)
        os_[starts + T - 1] = pred
        # autoregressive: after OBSERVED steps the model only sees its own predictions
        y_ar = y.copy()
        for p in range(OBSERVED, L):
            pred, _, _ = forward(m, workout_windows(rd, r, np.array([p - T + 1]), y_ar))
            y_ar[p] = pred[0]
        ar = np.full(L, np.nan)
        ar[OBSERVED:] = y_ar[OBSERVED:]
        true.append(y)
        one_step.append(os_)
        rollout.append(ar)
    np.savez(
        path,
        rows=np.array(rows),
        sport=c["sport_idx"][rows],
        true=np.array(true),
        one_step=np.array(one_step),
        rollout=np.array(rollout),
    )


def compute():
    os.makedirs(CACHE, exist_ok=True)
    fa = load_training_module()
    m = build_model(fa)
    sport_names = load_sport_names()
    compute_examples(fa, m, sport_names)
    compute_test(fa, m)
    compute_curves(fa, m)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def load_sport_names():
    sys.path.insert(0, HERE)  # metaDataEndomondo is defined in data_interpolate
    with open("data/endomondoHR_proper_metaData.pkl", "rb") as f:
        meta = pickle.load(f)
    smap = meta.oneHotMap["sport"]
    names = [None] * len(smap)
    for name, i in smap.items():
        names[i] = name
    return names


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )
    return plt


def fig_data_example(plt, c, sport_names):
    """Figure 1: one workout with its input channels, target and the context (previous) workout."""
    rows = c["test_rows"]
    run = sport_names.index("run")
    hr = c["seq"][rows][:, :, 2]
    spread = hr[:, 10:].std(axis=1)
    cand = rows[(c["sport_idx"][rows] == run) & (spread > np.percentile(spread, 75))]
    r = int(np.random.default_rng(SEED + 1).choice(cand))
    ctx = c["ctx_row"][r]
    x = np.arange(c["seq"].shape[1])

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), sharex=True)
    panels = [
        (0, "Velocidade\n(entrada, normalizada)", BLUE),
        (1, "Altitude\n(entrada, normalizada)", AQUA),
        (2, "Frequência cardíaca\n(alvo, bpm)", INK),
    ]
    for ax, (ch, label, color) in zip(axes, panels):
        ax.plot(x[1:], c["seq"][r, 1:, ch], color=color, label="treino atual")
        ax.plot(x[1:], c["seq"][ctx, 1:, ch], color=INK2, lw=1.2, ls="--", alpha=0.8, label="treino anterior (contexto)")
        ax.set_ylabel(label, fontsize=9)
    axes[0].legend(loc="lower right", ncol=2, fontsize=9)
    fig.align_ylabels(axes)
    axes[-1].set_xlabel("Passo de tempo (amostra do treino)")
    # illustrate one input window and its target
    w0 = 120
    for ax in axes:
        ax.axvspan(w0, w0 + T - 1, color=ORANGE, alpha=0.15, lw=0)
    axes[2].annotate(
        "janela de T = 10 passos:\nentradas + 9 FCs anteriores\n→ prevê a FC do 10º passo",
        xy=(w0 + T - 1, c["seq"][r, w0 + T - 1, 2]),
        xytext=(w0 + 40, np.nanmin(c["seq"][r, 1:, 2]) + 2),
        fontsize=8.5,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=INK2),
    )
    fig.suptitle(
        "Exemplo de treino do Endomondo (corrida, conjunto de teste) e o treino anterior do mesmo usuário",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG + "/fig1_exemplo_dados.png")
    plt.close(fig)


def fig_dataset(plt, c, sport_names):
    """Figure 2: what is in the dataset (sports, workouts per user, heart rate distribution)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    all_rows = np.concatenate([c["train_rows"], c["valid_rows"], c["test_rows"]])

    counts = np.bincount(c["sport_idx"][all_rows], minlength=len(sport_names))
    order = np.argsort(counts)[::-1]
    top = order[:5]
    labels = [sport_names[i] for i in top] + ["outros"]
    vals = list(counts[top]) + [counts[order[5:]].sum()]
    ax = axes[0]
    ax.barh(labels[::-1], vals[::-1], color=BLUE, height=0.6)
    ax.set_title("Treinos por esporte")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, " {:,}".format(v).replace(",", "."), va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, max(vals) * 1.25)

    per_user = np.bincount(c["user_idx"][all_rows])
    per_user = per_user[per_user > 0]
    ax = axes[1]
    ax.hist(per_user, bins=np.logspace(0, np.log10(per_user.max() + 1), 30), color=BLUE, edgecolor="white", lw=0.5)
    ax.set_xscale("log")
    ax.set_title("Treinos por usuário ({} usuários)".format(len(per_user)))
    ax.set_xlabel("nº de treinos (escala log)")

    hr = c["seq"][all_rows, 1:, 2].ravel()
    ax = axes[2]
    ax.hist(hr, bins=80, range=(50, 210), color=BLUE)
    ax.set_title("Frequência cardíaca (todas as amostras)")
    ax.set_xlabel("bpm")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG + "/fig2_dataset.png")
    plt.close(fig)


def fig_model(plt):
    """Figure 3: architecture diagram of FitRec-Attn (as configured in fitrect-attn.py)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 126)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#ffffff", ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.3))
        if sub:
            ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
            ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=8, color=INK2, linespacing=1.3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)

    def arrow(*pts):
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:-1]):
            ax.plot([x0, x1], [y0, y1], color=INK2, lw=1.2)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2, shrinkA=0, shrinkB=1))

    inp, ctx, enc, dec, out = "#eef4fc", "#f2f0fc", "#e8f6f0", "#fdf0ea", "#fff7e0"
    H = 8
    # inputs
    ax.text(2, 58.5, "Entradas", fontsize=10, fontweight="bold", color=INK)
    box(2, 48, 24, H, "Treino anterior", "velocidade, altitude, Δt", inp)
    box(2, 37, 24, H, "Treino anterior", "frequência cardíaca", inp)
    box(2, 24, 24, H, "Treino atual (10 passos)", "velocidade, altitude", inp)
    box(2, 13, 24, H, "Usuário", "userId", inp)
    box(2, 2, 24, H, "FC real", "9 passos anteriores", inp)

    # context encoder
    ax.text(33, 58.5, "Codificador de contexto (dropout 0.2)", fontsize=9, color=VIOLET, fontweight="bold")
    box(33, 48, 18, H, "LSTM 1", "hidden 64", ctx)
    box(33, 37, 18, H, "LSTM 2", "hidden 64", ctx)
    box(58, 42.5, 20, H, "Concat + Linear", "contexto (dim 32)", ctx)
    box(33, 13, 18, H, "Embedding", "usuário (dim 5)", ctx)
    arrow((26, 52), (33, 52))
    arrow((26, 41), (33, 41))
    arrow((51, 52), (58, 48.5))
    arrow((51, 41), (58, 44.5))
    arrow((26, 17), (33, 17))

    # encoder
    ax.text(58, 15, "Codificador (dropout 0.1)", fontsize=9, color=AQUA, fontweight="bold")
    box(57, 18, 23, 13, "Atenção nas\nentradas", "\n\npesa as 39 variáveis\npor passo (2 + 5 + 32)", enc)
    box(85, 18, 14, 13, "LSTM", "codificador\nhidden 64", enc)
    arrow((68, 42.5), (68, 31))
    arrow((26, 28), (57, 28))
    arrow((51, 17), (57, 21))
    arrow((80, 24.5), (85, 24.5))

    # decoder
    box(105, 34, 18, 10, "Atenção\ntemporal", "\n\npesa os 10 passos", dec)
    box(105, 17, 18, 11, "LSTM", "decodificador\nhidden 64", dec)
    box(105, 50, 18, H, "FC prevista", "passo 10 (bpm)", out, ORANGE)
    ax.text(109, 13.5, "Decodificador", fontsize=9, color=ORANGE, fontweight="bold")
    arrow((99, 28), (105, 38))
    arrow((114, 34), (114, 28))
    arrow((26, 6), (107, 6), (107, 17))
    arrow((123, 23), (125, 23), (125, 54), (123, 54))
    arrow((114, 44), (114, 50))
    ax.text(103, 47, "Linear(estado + contexto)", fontsize=7.5, color=INK2, ha="right")

    fig.suptitle("Arquitetura do FitRec-Attn (encoder-decoder com dupla atenção)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(FIG + "/fig3_modelo.png")
    plt.close(fig)


def fig_learning_curve(plt, curves, test):
    base_va = curves[min(curves, key=int)]["valid"]["baseline_mse"] if curves else None
    hist = load_json(RUN_DIR + "/history.json", [])
    if hist:
        # runs trained with the current fitrect-attn.py log every epoch: use that directly
        epochs = [h["epoch"] for h in hist]
        tr = [h["train_loss"] for h in hist]
        va = [h["valid_loss"] for h in hist]
    else:
        # older run: losses re-measured from the saved checkpoints (train on a subset, eval mode)
        epochs = sorted(int(e) for e in curves)
        tr = [curves[str(e)]["train"]["mse"] for e in epochs]
        va = [curves[str(e)]["valid"]["mse"] for e in epochs]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ys, color, label in ((tr, BLUE, "treino"), (va, ORANGE, "validação")):
        ax.plot(epochs, ys, "o", color=color, ms=6, label=label)
        for (e0, y0), (e1, y1) in zip(zip(epochs, ys), zip(epochs[1:], ys[1:])):
            # dashed where epochs are missing: we only know the end points
            ax.plot([e0, e1], [y0, y1], color=color, ls="-" if e1 - e0 == 1 else "--")
    if base_va is not None:
        ax.axhline(base_va, color=INK2, ls=":", lw=1.5)
        ax.text(epochs[-1], base_va, "baseline: repetir a última FC (validação)  ", ha="right", va="bottom", fontsize=8.5, color=INK2)
    ax.axvline(BEST_EPOCH, color=GRID, lw=6, zorder=0)
    ax.text(BEST_EPOCH, max(tr + va), " melhor época ({})".format(BEST_EPOCH), fontsize=8.5, color=INK2, va="top")
    ax.set_xticks(range(0, epochs[-1] + 1))
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE (bpm²)")
    ax.set_title("Curva de aprendizado")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88))
    missing = [e for e in range(epochs[-1]) if e not in epochs]
    if missing:
        ax.annotate(
            "épocas {} sem checkpoint salvo\n(não melhoraram a validação; early stopping na 7)".format(
                ", ".join(map(str, missing))
            ),
            xy=(np.mean(missing), max(va[-2:]) + 0.35),
            ha="center",
            fontsize=8,
            color=INK2,
        )
    fig.tight_layout()
    fig.savefig(FIG + "/fig4_curva_aprendizado.png")
    plt.close(fig)


def fig_predictions(plt, ex, sport_names, test):
    n = len(ex["rows"])
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.5 * n), sharex=True)
    x = np.arange(ex["true"].shape[1])
    for i, ax in enumerate(axes):
        y, os_, ar = ex["true"][i], ex["one_step"][i], ex["rollout"][i]
        ax.plot(x[1:], y[1:], color=INK, lw=1.6, label="FC real")
        ax.plot(x, os_, color=BLUE, lw=1.4, label="previsão 1 passo à frente")
        ax.plot(x, ar, color=ORANGE, lw=1.8, ls="--", label="continuação autoregressiva")
        ax.axvline(OBSERVED, color=INK2, lw=1, ls=":")
        m1 = np.nanmean(np.abs(os_[OBSERVED:] - y[OBSERVED:]))
        m2 = np.nanmean(np.abs(ar[OBSERVED:] - y[OBSERVED:]))
        ax.set_title(
            "{} (treino de teste #{})  —  MAE 1 passo: {:.1f} bpm  |  MAE autoregressivo: {:.1f} bpm".format(
                sport_names[ex["sport"][i]], ex["rows"][i], m1, m2
            ),
            fontsize=9.5,
        )
        ax.set_ylabel("bpm")
    axes[0].text(OBSERVED, axes[0].get_ylim()[0], " a partir daqui a curva laranja\n só vê as próprias previsões", fontsize=8, color=INK2, va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    axes[-1].set_xlabel("Passo de tempo")
    fig.suptitle("Frequência cardíaca real vs. prevista (melhor modelo, época {})".format(BEST_EPOCH), x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG + "/fig5_previsoes.png")
    plt.close(fig)


def fig_results(plt, test):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    labels = ["FitRec-Attn", "Repetir última FC"]
    for ax, key, title in zip(axes, ("rmse", "mae"), ("RMSE no teste (bpm)", "MAE no teste (bpm)")):
        vals = [test[key], test["baseline_" + key]]
        bars = ax.bar(labels, vals, color=[BLUE, "#b9b8b2"], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, "{:.2f}".format(v), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle("Resultado final no conjunto de teste ({:,} janelas)".format(test["n_windows"]).replace(",", "."), x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG + "/fig6_resultado_teste.png")
    plt.close(fig)


def plot():
    plt = setup_mpl()
    os.makedirs(FIG, exist_ok=True)
    sport_names = load_sport_names()
    with np.load("data/fitrec_compact.npz") as f:
        c = {k: f[k] for k in f.files}
    fig_data_example(plt, c, sport_names)
    fig_dataset(plt, c, sport_names)
    fig_model(plt)
    test = load_json(CACHE + "/test_metrics.json", None)
    curves = load_json(CACHE + "/curves.json", {})
    if curves or os.path.exists(RUN_DIR + "/history.json"):
        fig_learning_curve(plt, curves, test)
    if os.path.exists(CACHE + "/examples.npz"):
        with np.load(CACHE + "/examples.npz") as f:
            fig_predictions(plt, {k: f[k] for k in f.files}, sport_names, test)
    if test:
        fig_results(plt, test)
        print(json.dumps(test, indent=1))
    print("figures written to", os.path.abspath(FIG))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("compute", "all"):
        compute()
    if what in ("plot", "all"):
        plot()
