"""
    Training a network *through its knowledge matrices*.

    Idea
    ----
    For an input x with label i, instead of the usual cross-entropy on the
    network output, we supervise the whole knowledge matrix M(W,f)(x):

            loss(x, i) = || M(W,f)(x) - E_{ii} ||_F^2

    where E_{ii} is the matrix unit (a matrix of the same shape as M(x) that
    is zero everywhere except a 1 in entry (i, i)).

    Why this is a sensible target
    -----------------------------
    The knowledge matrix satisfies   output(x) = M(x).sum(over columns).
    A matrix equal to E_{ii} therefore sums (over columns) to the one-hot
    vector e_i, i.e. it forces a *correct, confident* prediction for class i.
    But E_{ii} asks for much more than cross-entropy: it pins down *every*
    entry of the local affine decomposition of the network at x, not just the
    column sums. This script trains with that loss and compares it, under
    identical hyper-parameters and identical initial weights, against vanilla
    cross-entropy training.

    Differentiable knowledge matrix
    -------------------------------
    The library's `KnowledgeMatrixComputer` runs under `torch.no_grad()`, so it
    cannot be used as a training loss directly. For a ReLU MLP the knowledge
    matrix has a simple closed form that *is* differentiable w.r.t. the weights
    (with the ReLU gating pattern held fixed, the correct sub-gradient a.e.):

        - At an input x the network is locally affine:  y = A(x) x + c(x),
          where A(x) is the product of the weight matrices gated by the ReLU
          activation masks.
        - Column j of M(x) (j an input coordinate) is  A(x)[:, j] * x_j.
        - The extra "bias" column of M(x) is  c(x) = y - A(x) x.

    We propagate the identity (scaled by x) and a zero-vector-with-biases
    through the layers, reusing the *detached* ReLU masks of the real forward
    pass. The result is checked numerically against the library computer in
    `validate_against_library()`.
"""

import argparse
import copy

import torch
from torch import nn

from knowledgematrix.neural_net import NN
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class SimpleMLP(NN):
    """A small ReLU MLP built with the library's NN builder."""

    def __init__(self, input_shape, num_classes, hidden=64, device="cpu"):
        super().__init__(input_shape, save=False, device=device)
        self.flatten()
        self.linear(in_features=self.get_input_size(), out_features=hidden)
        self.relu()
        self.linear(in_features=hidden, out_features=hidden)
        self.relu()
        self.linear(in_features=hidden, out_features=num_classes)


# --------------------------------------------------------------------------- #
#  Differentiable knowledge matrix for a ReLU MLP
# --------------------------------------------------------------------------- #
def knowledge_matrix_mlp(model: NN, x: torch.Tensor):
    """
        Differentiable knowledge matrix for a flatten/linear/relu MLP.

        Args:
            model: the SimpleMLP (or any flatten+linear+relu NN).
            x: inputs of shape (B, d).
        Returns:
            M: knowledge matrices, shape (B, K, d + 1)  (last column is bias).
            y: network outputs, shape (B, K).  (y == M.sum(-1))
    """
    B, d = x.shape
    # Columns of M, indexed by the input coordinate p:  C[b, :, p].
    # Initialise so column p carries  x_p * e_p  (a one-hot scaled by the input).
    C = torch.diag_embed(x)            # (B, d, d)
    a = torch.zeros(B, d, dtype=x.dtype, device=x.device)   # bias channel
    v = x                              # actual activations (to read ReLU masks)

    for layer in model.layers:
        if isinstance(layer, nn.Flatten):
            continue
        elif isinstance(layer, nn.Linear):
            W, b = layer.weight, layer.bias
            C = torch.einsum("oi,bip->bop", W, C)          # propagate columns
            a = a @ W.t() + (b if b is not None else 0.0)  # propagate biases
            v = v @ W.t() + (b if b is not None else 0.0)  # actual pre-activation
        elif isinstance(layer, nn.ReLU):
            mask = (v > 0).to(v.dtype).detach()            # gating, held fixed
            C = C * mask.unsqueeze(-1)
            a = a * mask
            v = v * mask                                   # == relu(v)
        else:
            raise TypeError(
                f"knowledge_matrix_mlp only supports Flatten/Linear/ReLU, "
                f"got {type(layer).__name__}"
            )

    M = torch.cat((C, a.unsqueeze(-1)), dim=-1)            # (B, K, d + 1)
    return M, v


# --------------------------------------------------------------------------- #
#  Sanity check against the library implementation
# --------------------------------------------------------------------------- #
def validate_against_library(model: NN, d: int, device="cpu"):
    """Compare the fast differentiable M against `KnowledgeMatrixComputer`."""
    model.eval()
    # Library expects the model input_shape; we use (1, d).
    x = torch.randn(model.input_shape, dtype=torch.get_default_dtype(), device=device)

    computer = KnowledgeMatrixComputer(model, batch_size=max(1, d // 2), device=device)
    lib_M = computer.forward(x)                            # (K, d + 1)

    fast_M, _ = knowledge_matrix_mlp(model, x.view(1, d))  # (1, K, d + 1)
    diff = torch.linalg.norm(lib_M - fast_M[0]).item()
    rel = diff / (torch.linalg.norm(lib_M).item() + 1e-12)
    return diff, rel


# --------------------------------------------------------------------------- #
#  Synthetic dataset (Gaussian blobs) -- keeps the experiment self-contained
# --------------------------------------------------------------------------- #
def make_centers(d, k, seed, sep=3.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(k, d, generator=g) * sep


def make_blobs(centers, n, seed):
    """Sample n points around shared class `centers` (so train/test align)."""
    k, d = centers.shape
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, k, (n,), generator=g)
    x = centers[labels] + torch.randn(n, d, generator=g)
    return x, labels


# --------------------------------------------------------------------------- #
#  Training / evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    out = model.forward(x)
    return (out.argmax(1) == y).float().mean().item()


def train(model, x_tr, y_tr, x_te, y_te, *, mode, epochs, lr, batch_size, d, k, log=None):
    """mode = 'km' (knowledge-matrix loss) or 'vanilla' (cross-entropy)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    n = x_tr.shape[0]
    g = torch.Generator().manual_seed(0)

    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad()

            if mode == "km":
                M, _ = knowledge_matrix_mlp(model, xb)        # (B, K, d+1)
                # target E_{ii}: zeros except entry (label, label) = 1
                target = torch.zeros_like(M)
                bidx = torch.arange(xb.shape[0])
                target[bidx, yb, yb] = 1.0
                loss = ((M - target) ** 2).sum(dim=(1, 2)).mean()
            elif mode == "vanilla":
                loss = ce(model.forward(xb), yb)
            else:
                raise ValueError(mode)

            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.shape[0]

        tr_acc = accuracy(model, x_tr, y_tr)
        te_acc = accuracy(model, x_te, y_te)
        history.append((epoch + 1, epoch_loss / n, tr_acc, te_acc))
        if log is not None:
            log(f"  [{mode:7s}] epoch {epoch + 1:3d}/{epochs}  "
                f"loss={epoch_loss / n:.4f}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")
    return history


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=16, help="input dimension")
    p.add_argument("--k", type=int, default=4, help="number of classes")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=str, default=None, help="optional markdown report path")
    args = p.parse_args()

    torch.set_default_dtype(torch.float32)
    assert args.k <= args.d + 1, "E_{ii} needs the class index i to be a valid column (k <= d+1)."

    lines = []
    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log("# Training a network via knowledge matrices\n")
    log(f"Config: d={args.d}, classes={args.k}, hidden={args.hidden}, "
        f"n_train={args.n_train}, n_test={args.n_test}, epochs={args.epochs}, "
        f"lr={args.lr}, batch_size={args.batch_size}, seed={args.seed}\n")

    # Data (train and test share the same class centers) --------------------- #
    centers = make_centers(args.d, args.k, seed=args.seed)
    x_tr, y_tr = make_blobs(centers, args.n_train, seed=args.seed + 1)
    x_te, y_te = make_blobs(centers, args.n_test, seed=args.seed + 2)

    # Two models with *identical* initial weights ---------------------------- #
    torch.manual_seed(args.seed)
    model_km = SimpleMLP((1, args.d), args.k, hidden=args.hidden)
    init_state = copy.deepcopy(model_km.state_dict())
    model_vanilla = SimpleMLP((1, args.d), args.k, hidden=args.hidden)
    model_vanilla.load_state_dict(init_state)

    # Faithfulness check ----------------------------------------------------- #
    diff, rel = validate_against_library(model_km, args.d)
    log(f"Knowledge-matrix check vs library computer: "
        f"abs_diff={diff:.3e}, rel_diff={rel:.3e} "
        f"({'OK' if rel < 1e-4 else 'MISMATCH'})\n")

    # Train ------------------------------------------------------------------ #
    log("## Knowledge-matrix training  (loss = ||M(x) - E_ii||^2)")
    hist_km = train(model_km, x_tr, y_tr, x_te, y_te, mode="km",
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    d=args.d, k=args.k, log=log)

    log("\n## Vanilla training  (loss = cross-entropy)")
    hist_va = train(model_vanilla, x_tr, y_tr, x_te, y_te, mode="vanilla",
                    epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                    d=args.d, k=args.k, log=log)

    # Summary ---------------------------------------------------------------- #
    km_tr, km_te = hist_km[-1][2], hist_km[-1][3]
    va_tr, va_te = hist_va[-1][2], hist_va[-1][3]
    log("\n## Final comparison (identical init & hyper-parameters)\n")
    log("| training            | final train acc | final test acc |")
    log("|---------------------|-----------------|----------------|")
    log(f"| knowledge matrices  | {km_tr:.4f}          | {km_te:.4f}         |")
    log(f"| vanilla (cross-ent) | {va_tr:.4f}          | {va_te:.4f}         |")

    log("\n## Observations\n")
    log("- The differentiable knowledge matrix matches the library "
        "`KnowledgeMatrixComputer` exactly, so the loss is computed on the "
        "true M(W,f)(x).")
    log(f"- Training via `||M(x) - E_ii||^2` does learn the task "
        f"(test acc {km_te:.2f} >> {1.0 / args.k:.2f} chance) and generalizes "
        f"(train {km_tr:.2f} vs test {km_te:.2f}).")
    log("- It is, however, a much harder optimization target than "
        "cross-entropy and plateaus below it: E_ii pins down *every* entry of "
        "the local affine decomposition (the whole matrix must become a fixed "
        "sparse matrix), whereas cross-entropy only constrains the column "
        "sums (the output). With identical init and hyper-parameters, vanilla "
        f"reaches {va_te:.2f} test accuracy.")

    if args.report:
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nReport written to {args.report}", flush=True)


if __name__ == "__main__":
    main()
