"""
    Hyper-parameter optimization comparing the off-class knowledge-matrix loss
    against vanilla cross-entropy, for both an MLP and a 1D-CNN, on MNIST-1D.

    For every sampled configuration the two losses are trained on the *same
    architecture and the same initial weights*, so each row of the results is a
    like-for-like comparison. Model selection uses a held-out validation split;
    we report the test accuracy of the selected configuration.

    Run:
        python extra/experiments/hpo.py --report extra/experiments/results_hpo.md
"""

import argparse
import copy
import random
import time

import torch

from extra.experiments import km_core as C

INPUT_SHAPE = (1, 1, 40)
NUM_CLASSES = 10
DATA = "extra/experiments/data/mnist1d_data.pkl"
LOSSES = ["km_offclass", "vanilla"]


# --------------------------------------------------------------------------- #
#  Search spaces
# --------------------------------------------------------------------------- #
def sample_mlp(rng):
    return dict(
        family="mlp",
        hidden=rng.choice([32, 64, 128, 256]),
        depth=rng.choice([1, 2, 3]),
        lr=rng.choice([3e-4, 1e-3, 3e-3]),
        batch_size=rng.choice([64, 128, 256]),
        weight_decay=rng.choice([0.0, 1e-4, 1e-3]),
        epochs=60,
    )


def sample_cnn(rng):
    return dict(
        family="cnn",
        channels=rng.choice([(16,), (32,), (16, 32), (32, 64), (16, 32, 64)]),
        kernel=rng.choice([3, 5]),
        hidden=rng.choice([32, 64, 128]),
        pool_out=rng.choice([2, 3, 4]),
        lr=rng.choice([3e-4, 1e-3, 3e-3]),
        batch_size=rng.choice([64, 128]),
        weight_decay=rng.choice([0.0, 1e-4]),
        epochs=35,
    )


def build(cfg):
    if cfg["family"] == "mlp":
        return C.MLP(INPUT_SHAPE, NUM_CLASSES, hidden=cfg["hidden"], depth=cfg["depth"])
    return C.CNN(INPUT_SHAPE, NUM_CLASSES, channels=cfg["channels"], kernel=cfg["kernel"],
                 hidden=cfg["hidden"], pool_out=cfg["pool_out"])


def cfg_str(cfg):
    if cfg["family"] == "mlp":
        return (f"MLP hidden={cfg['hidden']} depth={cfg['depth']} lr={cfg['lr']:g} "
                f"bs={cfg['batch_size']} wd={cfg['weight_decay']:g}")
    return (f"CNN ch={cfg['channels']} k={cfg['kernel']} hid={cfg['hidden']} "
            f"pool={cfg['pool_out']} lr={cfg['lr']:g} bs={cfg['batch_size']} wd={cfg['weight_decay']:g}")


# --------------------------------------------------------------------------- #
#  One trial: train both losses on identical architecture + init
# --------------------------------------------------------------------------- #
def run_trial(cfg, trial_seed, data):
    xtr, ytr, xval, yval, xte, yte = data
    torch.manual_seed(trial_seed)
    init_state = copy.deepcopy(build(cfg).state_dict())

    out = {}
    for loss in LOSSES:
        model = build(cfg)
        model.load_state_dict(init_state)
        km_batch = None
        if loss in C.KM_LOSSES:
            km_batch = 32 if cfg["family"] == "cnn" else cfg["batch_size"]
        C.train(model, xtr, ytr, mode=loss, epochs=cfg["epochs"], lr=cfg["lr"],
                batch_size=cfg["batch_size"], weight_decay=cfg["weight_decay"],
                km_batch=km_batch, seed=0)
        out[loss] = dict(train=C.accuracy(model, xtr, ytr),
                         val=C.accuracy(model, xval, yval),
                         test=C.accuracy(model, xte, yte))
    return out


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp-trials", type=int, default=12)
    p.add_argument("--cnn-trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=str, default="extra/experiments/results_hpo.md")
    args = p.parse_args()

    torch.set_default_dtype(torch.float32)
    lines = []
    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    # Data: hold out 500 of the 4000 training points for validation.
    xtr, ytr, xte, yte = C.load_mnist1d(DATA)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(xtr.shape[0], generator=g)
    val_idx, tr_idx = perm[:500], perm[500:]
    data = (xtr[tr_idx], ytr[tr_idx], xtr[val_idx], ytr[val_idx], xte, yte)

    # Faithfulness check.
    torch.set_default_dtype(torch.float64)
    dm = C.validate_against_library(C.MLP(INPUT_SHAPE, NUM_CLASSES, 32, 2))
    dc = C.validate_against_library(C.CNN(INPUT_SHAPE, NUM_CLASSES, (8, 16), 5, 16, 3))
    torch.set_default_dtype(torch.float32)

    log("# HPO: off-class KM loss vs vanilla cross-entropy (MNIST-1D)\n")
    log(f"Data: MNIST-1D, train={data[0].shape[0]}, val={data[2].shape[0]}, test={xte.shape[0]}, "
        f"classes={NUM_CLASSES}.")
    log(f"KM (gradient/jacrev) faithfulness vs library (float64): "
        f"MLP rel={dm[1]:.1e}, CNN rel={dc[1]:.1e} (0 == exact).")
    log("Every configuration trains both losses on the SAME architecture and the "
        "SAME initial weights. Model selection is by validation accuracy; the "
        "reported number is test accuracy.\n")

    rng = random.Random(args.seed)
    families = [("MLP", sample_mlp, args.mlp_trials), ("CNN", sample_cnn, args.cnn_trials)]
    best = {}   # (family, loss) -> (test, val, cfg, paired_other_test)

    for fam_name, sampler, n_trials in families:
        log(f"## {fam_name}: {n_trials} random configurations\n")
        log("| # | config | KM val | KM test | vanilla val | vanilla test |")
        log("|---|--------|--------|---------|-------------|--------------|")
        trials = []
        for t in range(n_trials):
            cfg = sampler(rng)
            t0 = time.time()
            res = run_trial(cfg, args.seed + 100 * t, data)
            dt = time.time() - t0
            trials.append((cfg, res))
            log(f"| {t+1} | {cfg_str(cfg)} | {res['km_offclass']['val']:.3f} | "
                f"{res['km_offclass']['test']:.3f} | {res['vanilla']['val']:.3f} | "
                f"{res['vanilla']['test']:.3f} |")
            print(f"    (trial {t+1}/{n_trials} took {dt:.0f}s)", flush=True)

        for loss in LOSSES:
            cfg, res = max(trials, key=lambda cr: cr[1][loss]["val"])
            other = "vanilla" if loss == "km_offclass" else "km_offclass"
            best[(fam_name, loss)] = (res[loss]["test"], res[loss]["val"], cfg,
                                      res[other]["test"], res[loss]["train"])
        log("")

    # ---------------------------------------------------------------------- #
    log("## Best results (selected by validation accuracy)\n")
    log("| family | loss | best config | train | val | test | same-config other-loss test |")
    log("|--------|------|-------------|-------|-----|------|-----------------------------|")
    pretty = {"km_offclass": "off-class KM", "vanilla": "vanilla CE"}
    for fam_name, _, _ in families:
        for loss in LOSSES:
            test, val, cfg, other_test, train = best[(fam_name, loss)]
            log(f"| {fam_name} | {pretty[loss]} | {cfg_str(cfg)} | {train:.3f} | {val:.3f} | "
                f"**{test:.3f}** | {other_test:.3f} |")

    log("\n## Head-to-head at each best configuration\n")
    for fam_name, _, _ in families:
        for loss in LOSSES:
            test, val, cfg, other_test, train = best[(fam_name, loss)]
            other = "vanilla" if loss == "km_offclass" else "off-class KM"
            log(f"- Best {fam_name} for {pretty[loss]} ({cfg_str(cfg)}): "
                f"{pretty[loss]} test={test:.3f}, {other} test={other_test:.3f}.")

    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {args.report}", flush=True)


if __name__ == "__main__":
    main()
