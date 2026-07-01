"""
    Regularized HPO: off-class KM loss vs vanilla cross-entropy on MNIST-1D,
    searching over architecture (depth x width), weight decay, and the LR
    scheduler, with SGD + Nesterov momentum.

    "Mid-training best counts": every configuration is evaluated on the
    validation and test sets each epoch, and we track the best-by-validation
    checkpoint (early-stopping accuracy) in addition to the final-epoch value.
    Configurations are selected by their best-val accuracy; we report the test
    accuracy at that checkpoint (and, for reference, the final-epoch test).

    All layers are ReLU/affine/pool, so the differentiable gradient (jacrev)
    knowledge matrix applies.

    Run:
        python extra/experiments/hpo_reg.py --report extra/experiments/results_reg.md
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

WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
SCHEDULERS = ["none", "cosine", "step", "exp", "onecycle", "plateau"]
LRS = [1e-2, 3e-2, 1e-1, 3e-1]


def sample_mlp(rng, epochs):
    return dict(family="mlp", depth=rng.choice([1, 2, 3, 4]),
                width=rng.choice([32, 64, 128, 256, 512]),
                lr=rng.choice(LRS), batch_size=rng.choice([64, 128, 256]),
                weight_decay=rng.choice(WEIGHT_DECAYS),
                scheduler=rng.choice(SCHEDULERS), epochs=epochs)


def sample_cnn(rng, epochs):
    return dict(family="cnn", depth=rng.choice([1, 2, 3]),
                width=rng.choice([8, 16, 32]), kernel=rng.choice([3, 5]),
                lr=rng.choice(LRS), batch_size=rng.choice([64, 128]),
                weight_decay=rng.choice(WEIGHT_DECAYS),
                scheduler=rng.choice(SCHEDULERS), epochs=epochs)


def build(cfg):
    if cfg["family"] == "mlp":
        return C.MLP(INPUT_SHAPE, NUM_CLASSES, hidden=cfg["width"], depth=cfg["depth"])
    channels = tuple(cfg["width"] * (2 ** i) for i in range(cfg["depth"]))
    return C.CNN(INPUT_SHAPE, NUM_CLASSES, channels=channels, kernel=cfg["kernel"],
                 hidden=64, pool_out=2)


def cfg_str(cfg):
    if cfg["family"] == "mlp":
        base = f"MLP d={cfg['depth']} w={cfg['width']}"
    else:
        ch = tuple(cfg["width"] * (2 ** i) for i in range(cfg["depth"]))
        base = f"CNN d={cfg['depth']} ch={ch} k={cfg['kernel']}"
    return f"{base} lr={cfg['lr']:g} bs={cfg['batch_size']} wd={cfg['weight_decay']:g} sched={cfg['scheduler']}"


def run_trial(cfg, trial_seed, data, opt_cfg):
    xtr, ytr, xval, yval, xte, yte = data
    torch.manual_seed(trial_seed)
    init_state = copy.deepcopy(build(cfg).state_dict())
    eval_data = (xval, yval, xte, yte)
    out = {}
    for loss in LOSSES:
        model = build(cfg)
        model.load_state_dict(init_state)
        km_batch = None
        if loss in C.KM_LOSSES:
            km_batch = 32 if cfg["family"] == "cnn" else cfg["batch_size"]
        m = C.train(model, xtr, ytr, mode=loss, epochs=cfg["epochs"], lr=cfg["lr"],
                    batch_size=cfg["batch_size"], weight_decay=cfg["weight_decay"],
                    km_batch=km_batch, seed=0, scheduler=cfg["scheduler"],
                    eval_data=eval_data, eval_every=1, return_metrics=True, **opt_cfg)
        out[loss] = m
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlp-trials", type=int, default=16)
    p.add_argument("--cnn-trials", type=int, default=12)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=str, default="extra/experiments/results_reg.md")
    args = p.parse_args()

    torch.set_default_dtype(torch.float32)
    opt_cfg = dict(optimizer="sgd", momentum=args.momentum)
    lines = []
    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    xtr, ytr, xte, yte = C.load_mnist1d(DATA)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(xtr.shape[0], generator=g)
    val_idx, tr_idx = perm[:500], perm[500:]
    data = (xtr[tr_idx], ytr[tr_idx], xtr[val_idx], ytr[val_idx], xte, yte)

    log("# Regularized HPO: off-class KM vs vanilla, MNIST-1D\n")
    log(f"Data: train={data[0].shape[0]}, val={data[2].shape[0]}, test={xte.shape[0]}, "
        f"classes={NUM_CLASSES}.")
    log(f"Search: architecture (depth x width), weight_decay {WEIGHT_DECAYS}, "
        f"LR scheduler {SCHEDULERS}, lr {LRS}. Optimizer: SGD(momentum={args.momentum}, "
        f"Nesterov), {args.epochs} epochs. All layers ReLU (gradient KM applies).")
    log("Both losses share architecture + init per config. Selection is by the "
        "**best-by-validation checkpoint over training** (mid-training best); we "
        "report that checkpoint's test accuracy and the final-epoch test.\n")

    rng = random.Random(args.seed)
    families = [("MLP", sample_mlp, args.mlp_trials), ("CNN", sample_cnn, args.cnn_trials)]
    best = {}
    sched_scores = {}   # (family, loss, scheduler) -> list of best-val test

    for fam, sampler, n_trials in families:
        log(f"## {fam}: {n_trials} random configurations\n")
        log("| # | config | KM best-test | KM final | van best-test | van final |")
        log("|---|--------|--------------|----------|---------------|-----------|")
        trials = []
        for t in range(n_trials):
            cfg = sampler(rng, args.epochs)
            t0 = time.time()
            res = run_trial(cfg, args.seed + 100 * t, data, opt_cfg)
            trials.append((cfg, res))
            km, va = res["km_offclass"], res["vanilla"]
            log(f"| {t+1} | {cfg_str(cfg)} | {km['best']['test']:.3f} | {km['final']['test']:.3f} "
                f"| {va['best']['test']:.3f} | {va['final']['test']:.3f} |")
            for loss in LOSSES:
                sched_scores.setdefault((fam, loss, cfg["scheduler"]), []).append(res[loss]["best"]["test"])
            print(f"    (trial {t+1}/{n_trials} {cfg['scheduler']} took {time.time()-t0:.0f}s)", flush=True)

        for loss in LOSSES:
            cfg, res = max(trials, key=lambda cr: cr[1][loss]["best"]["val"])
            b = res[loss]["best"]
            best[(fam, loss)] = (cfg, b["epoch"], b["val"], b["test"],
                                 res[loss]["final"]["test"], res[loss]["final"]["train"])
        log("")

    log("## Best configuration in every case (selected by mid-training best val)\n")
    log("| family | loss | best config | best epoch | val | **best test** | final test |")
    log("|--------|------|-------------|-----------|-----|---------------|------------|")
    pretty = {"km_offclass": "off-class KM", "vanilla": "vanilla CE"}
    for fam, _, _ in families:
        for loss in LOSSES:
            cfg, ep, vl, te, fte, ftr = best[(fam, loss)]
            log(f"| {fam} | {pretty[loss]} | {cfg_str(cfg)} | {ep} | {vl:.3f} | "
                f"**{te:.3f}** | {fte:.3f} |")

    log("\n## Which LR scheduler is best (mean best-test over its trials)\n")
    log("| family | loss | " + " | ".join(SCHEDULERS) + " |")
    log("|--------|------|" + "----|" * len(SCHEDULERS))
    for fam, _, _ in families:
        for loss in LOSSES:
            cells = []
            for s in SCHEDULERS:
                vals = sched_scores.get((fam, loss, s))
                cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "—")
            log(f"| {fam} | {pretty[loss]} | " + " | ".join(cells) + " |")

    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {args.report}", flush=True)


if __name__ == "__main__":
    main()
