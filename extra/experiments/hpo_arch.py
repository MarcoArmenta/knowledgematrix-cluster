"""
    Architecture sweep: which depth x width wins, for each model family and each
    loss (off-class knowledge-matrix vs vanilla cross-entropy), on MNIST-1D.

    Depth and width are the searched axes; the rest of the recipe is fixed and
    shared (SGD + momentum + cosine LR, 100 epochs) so the architecture is the
    controlled variable. Both losses are trained on the SAME architecture and
    SAME initial weights at every grid point. All layers are ReLU/affine/pool,
    so the differentiable gradient (jacrev) knowledge matrix applies.

    Run:
        python extra/experiments/hpo_arch.py --report extra/experiments/results_arch.md
"""

import argparse
import copy
import time

import torch

from extra.experiments import km_core as C

INPUT_SHAPE = (1, 1, 40)
NUM_CLASSES = 10
DATA = "extra/experiments/data/mnist1d_data.pkl"
LOSSES = ["km_offclass", "vanilla"]

# Architecture axes (depth x width).
MLP_DEPTHS = [1, 2, 3, 4]
MLP_WIDTHS = [32, 64, 128, 256]
CNN_DEPTHS = [1, 2, 3]              # number of conv blocks
CNN_WIDTHS = [8, 16, 32]           # base channels; block i has width * 2**i


def build_mlp(depth, width):
    return C.MLP(INPUT_SHAPE, NUM_CLASSES, hidden=width, depth=depth)


def build_cnn(depth, width):
    channels = tuple(width * (2 ** i) for i in range(depth))
    return C.CNN(INPUT_SHAPE, NUM_CLASSES, channels=channels, kernel=5,
                 hidden=64, pool_out=2)


def train_eval(make, arch_seed, data, epochs, opt_cfg, lr, bs, wd, km_batch):
    xtr, ytr, xval, yval, xte, yte = data
    torch.manual_seed(arch_seed)
    init_state = copy.deepcopy(make().state_dict())
    out = {}
    for loss in LOSSES:
        model = make()
        model.load_state_dict(init_state)
        kmb = km_batch if loss in C.KM_LOSSES else None
        C.train(model, xtr, ytr, mode=loss, epochs=epochs, lr=lr, batch_size=bs,
                weight_decay=wd, km_batch=kmb, seed=0, **opt_cfg)
        out[loss] = dict(train=C.accuracy(model, xtr, ytr),
                         val=C.accuracy(model, xval, yval),
                         test=C.accuracy(model, xte, yte))
    return out


def grid_table(results, depths, widths, loss, key="test"):
    lines = [f"| depth \\ width | " + " | ".join(str(w) for w in widths) + " |",
             "|" + "---|" * (len(widths) + 1)]
    for dpt in depths:
        cells = [f"{results[(dpt, w)][loss][key]:.3f}" for w in widths]
        lines.append(f"| **{dpt}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gap_table(results, depths, widths, loss):
    lines = [f"| depth \\ width | " + " | ".join(str(w) for w in widths) + " |",
             "|" + "---|" * (len(widths) + 1)]
    for dpt in depths:
        cells = []
        for w in widths:
            r = results[(dpt, w)][loss]
            cells.append(f"{r['train'] - r['test']:.3f}")
        lines.append(f"| **{dpt}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=str, default="extra/experiments/results_arch.md")
    args = p.parse_args()

    torch.set_default_dtype(torch.float32)
    opt_cfg = dict(optimizer=args.optimizer, momentum=args.momentum, scheduler=args.scheduler)
    lines = []
    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    # Data: hold out 500 of 4000 training points for validation.
    xtr, ytr, xte, yte = C.load_mnist1d(DATA)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(xtr.shape[0], generator=g)
    val_idx, tr_idx = perm[:500], perm[500:]
    data = (xtr[tr_idx], ytr[tr_idx], xtr[val_idx], ytr[val_idx], xte, yte)

    log("# Architecture sweep (depth x width): off-class KM vs vanilla, MNIST-1D\n")
    opt_txt = (f"SGD (momentum={args.momentum}, Nesterov)" if args.optimizer == "sgd" else "Adam")
    log(f"Fixed recipe: {opt_txt}, "
        f"{'cosine LR' if args.scheduler == 'cosine' else 'constant LR'}, "
        f"{args.epochs} epochs, lr={args.lr:g}, batch={args.batch_size}, wd={args.weight_decay:g}. "
        "All layers ReLU/affine/pool (gradient KM applies).")
    log(f"Data: MNIST-1D, train={data[0].shape[0]}, val={data[2].shape[0]}, "
        f"test={xte.shape[0]}, classes={NUM_CLASSES}. Both losses share architecture "
        "and initial weights at every grid point; winners selected by validation accuracy.\n")

    families = [
        ("MLP", build_mlp, MLP_DEPTHS, MLP_WIDTHS, args.batch_size, "hidden width"),
        ("CNN", build_cnn, CNN_DEPTHS, CNN_WIDTHS, 32, "base channels"),
    ]
    winners = {}

    for fam, make_fn, depths, widths, km_batch, width_name in families:
        log(f"## {fam}: depth x {width_name} grid  ({len(depths)}x{len(widths)} architectures)\n")
        results = {}
        for dpt in depths:
            for w in widths:
                t0 = time.time()
                res = train_eval(lambda d=dpt, ww=w: make_fn(d, ww),
                                 args.seed + 1000 * dpt + w, data,
                                 args.epochs, opt_cfg, args.lr, args.batch_size,
                                 args.weight_decay, km_batch)
                results[(dpt, w)] = res
                print(f"    {fam} depth={dpt} width={w}: "
                      f"KM test={res['km_offclass']['test']:.3f} "
                      f"vanilla test={res['vanilla']['test']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        for loss in LOSSES:
            name = "off-class KM" if loss == "km_offclass" else "vanilla CE"
            log(f"### {fam} — {name}: test accuracy\n")
            log(grid_table(results, depths, widths, loss, "test"))
            log("")
            log(f"### {fam} — {name}: train accuracy\n")
            log(grid_table(results, depths, widths, loss, "train"))
            log("")
            log(f"### {fam} — {name}: generalization gap (train − test)\n")
            log(gap_table(results, depths, widths, loss))
            log("")
            # winner by validation accuracy
            (bd, bw) = max(results, key=lambda k: results[k][loss]["val"])
            r = results[(bd, bw)][loss]
            winners[(fam, loss)] = (bd, bw, r["train"], r["val"], r["test"])

    log("## Winning architecture in every case (selected by validation accuracy)\n")
    log("| family | loss | winning architecture | train | val | test | gap (train−test) |")
    log("|--------|------|----------------------|-------|-----|------|------------------|")
    for fam, _, _, _, _, wn in families:
        for loss in LOSSES:
            name = "off-class KM" if loss == "km_offclass" else "vanilla CE"
            bd, bw, tr, vl, te = winners[(fam, loss)]
            if fam == "MLP":
                arch = f"depth={bd}, width={bw}"
            else:
                ch = tuple(bw * (2 ** i) for i in range(bd))
                arch = f"depth={bd}, channels={ch}"
            log(f"| {fam} | {name} | {arch} | {tr:.3f} | {vl:.3f} | **{te:.3f}** | {tr - te:.3f} |")

    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {args.report}", flush=True)


if __name__ == "__main__":
    main()
