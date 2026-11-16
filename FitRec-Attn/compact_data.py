"""One-time conversion: data/processed_endomondoHR_proper_interpolate.npy (+ split .pkl + metadata .pkl)
-> data/fitrec_compact.npz, which is all fitrect-attn.py needs.

Needs ~24 GB of RAM once (to read the original .npy). After that, training loads the compact
file (~1 GB) in seconds, so it also runs on machines with less RAM, e.g. Google Colab.

    python compact_data.py            # convert
    python compact_data.py --verify   # convert and check batches against the original code
"""
import argparse
import itertools
import os
import sys

import numpy as np

from data_interpolate import dataInterpreter, COMPACT_FN

parser = argparse.ArgumentParser()
parser.add_argument("--verify", action="store_true", help="compare compact batches with the original generator")
args = parser.parse_args()

if os.path.exists("data/" + COMPACT_FN):
    sys.exit("data/{} already exists; delete it to rebuild".format(COMPACT_FN))

# same settings as fitrect-attn.py
T = 10
reader = dataInterpreter(T, ['derived_speed', 'altitude'], True, False, True, ['heart_rate'],
                         fn="endomondoHR_proper.json", scaleVals=True, trimmed_workout_len=300,
                         scaleTargets=False, trainValidTestSplit=[0.8, 0.1, 0.1], zMultiple=5,
                         trainValidTestFN="data/endomondoHR_proper_temporal_dataset.pkl")
reader.preprocess_data(keep_original=args.verify)

if not args.verify:
    sys.exit(0)

KEYS = ['input', 'user_input', 'context_input_1', 'context_input_2', 'workoutid']

def check(name, old, new):
    oi, oo = old
    ni, no = new
    # the original code builds float64 arrays that the model casts to float32 (or to long for ids),
    # so compare after casting the old array to the new array's dtype
    pairs = [(oi[k], ni[k]) for k in KEYS] + [(oo, no)]
    for k, (a, b) in zip(KEYS + ['target'], pairs):
        b = np.asarray(b)
        a = np.asarray(a).astype(b.dtype)
        if a.shape != b.shape or not np.array_equal(a, b):
            sys.exit("MISMATCH in {} / {}: shapes {} vs {}".format(name, k, a.shape, b.shape))

batch_size = 12800
n_check = 3
for split in ["train", "valid", "test"]:
    old_gen = reader.legacy_generator_for_autotrain(batch_size, 300, split)
    new_gen = reader.generator_for_autotrain(batch_size, 300, split)
    for i, (old, new) in enumerate(itertools.islice(zip(old_gen, new_gen), n_check)):
        check("{} batch {}".format(split, i), old, new)
    print("{}: first {} batches identical, {} batches per epoch".format(split, n_check, reader.num_batches(split, batch_size)))

# spot-check random windows deep inside the epoch directly against the original dicts
rng = np.random.default_rng(0)
data = reader.original_data
for split, ids in [("train", reader.trainingSet), ("valid", reader.validationSet), ("test", reader.testSet)]:
    rows, starts, _ = reader._split_windows(split)
    n_batches = reader.num_batches(split, batch_size)
    for i in rng.integers(0, n_batches, size=20):
        inp, out = reader._build_batch(rows, starts, int(i), batch_size)
        for j in rng.integers(0, batch_size, size=50):
            w = int(np.searchsorted(starts, int(i) * batch_size + j, side='right') - 1)
            t = int(inp['workoutid'][j, 1])
            d = data[ids[w]]
            ctx = data[reader.contextMap[ids[w]][2][-1]]
            sl = slice(t, t + T)
            exp_in = np.stack([np.asarray(d[a], dtype=np.float64)[sl] for a in ['derived_speed', 'altitude']], 1)
            exp_out = np.asarray(d['tar_heart_rate'], dtype=np.float64)[sl]
            exp_c1 = np.stack([np.asarray(ctx[a], dtype=np.float64)[sl] for a in ['derived_speed', 'altitude']]
                              + [np.full(T, reader.contextMap[ids[w]][0], dtype=np.float64)], 1)
            exp_c2 = np.asarray(ctx['tar_heart_rate'], dtype=np.float64)[sl]
            ok = (np.array_equal(inp['input'][j], exp_in.astype(np.float32))
                  and np.array_equal(out[j, :, 0], exp_out.astype(np.float32))
                  and np.array_equal(inp['context_input_1'][j], exp_c1.astype(np.float32))
                  and np.array_equal(inp['context_input_2'][j, :, 0], exp_c2.astype(np.float32))
                  and inp['workoutid'][j, 0] == d['id']
                  and (inp['user_input'][j] == reader.oneHotMap['userId'][d['userId']]).all())
            if not ok:
                sys.exit("MISMATCH in {} batch {} row {}".format(split, i, j))
    print("{}: 1000 random windows identical".format(split))

print("VERIFY OK")
