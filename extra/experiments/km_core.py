"""
    Shared core for the knowledge-matrix training experiments.

    Provides
    --------
    * knowledge_matrix(model, x) : a *differentiable*, batched knowledge matrix
      for networks built from Flatten / Linear / ReLU / Conv2d / MaxPool2d /
      AvgPool2d / AdaptiveAvgPool2d (a superset of what the MLP experiment
      needed, so the same code covers CNNs). It is checked against the library's
      `KnowledgeMatrixComputer` in `validate_against_library`.
    * the knowledge-matrix losses (E_ii, off-class, row-norm variants).
    * data loaders (MNIST-1D, synthetic blobs) and model builders (MLP, CNN).
    * a generic train / accuracy loop.

    Design note (differentiability)
    --------------------------------
    The library computer runs under `torch.no_grad()`. For piecewise-linear nets
    the knowledge matrix has a closed form that IS differentiable w.r.t. the
    weights once the ReLU gating and max-pool argmax are fixed (the correct
    sub-gradient a.e.). We propagate, per input coordinate `p`, the one-hot input
    `x_p * e_p` through the layers (the "columns" of M) plus a zero-input-with-
    biases channel (the bias column), reusing the *detached* masks/argmax of the
    real forward pass. The input-coordinate axis is folded into the batch so a
    whole minibatch is done at once.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import jacrev, vmap

from knowledgematrix.neural_net import NN
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer


# --------------------------------------------------------------------------- #
#  Differentiable knowledge matrix (general: MLP and CNN layers)
# --------------------------------------------------------------------------- #
def knowledge_matrix(model: NN, x: torch.Tensor):
    """
        Args:
            model: an NN built from Flatten/Linear/ReLU/Conv2d/MaxPool2d/
                   AvgPool2d/AdaptiveAvgPool2d (no BatchNorm/residuals).
            x:     inputs of shape (B, C0, H0, W0).
        Returns:
            M: knowledge matrices, shape (B, K, d + 1)   (d = C0*H0*W0).
            y: network outputs,     shape (B, K).        (y == M.sum(-1))
    """
    B = x.shape[0]
    C0, H0, W0 = x.shape[1:]
    d = C0 * H0 * W0

    # P[b, p] = the contribution carried by input coordinate p: x_p * e_p.
    eye = torch.eye(d, dtype=x.dtype, device=x.device)          # (d, d)
    xf = x.reshape(B, d)                                        # (B, d)
    P = (eye.unsqueeze(0) * xf.unsqueeze(1)).reshape(B, d, C0, H0, W0)
    a = torch.zeros(B, C0, H0, W0, dtype=x.dtype, device=x.device)   # bias column
    v = x                                                      # actual activations

    for layer in model.layers:
        if isinstance(layer, (nn.Dropout, nn.Identity)):
            continue

        elif isinstance(layer, nn.Flatten):
            P = P.reshape(B, d, -1)
            a = a.reshape(B, -1)
            v = v.reshape(B, -1)

        elif isinstance(layer, nn.Linear):
            Wt = layer.weight.t()
            bias = layer.bias if layer.bias is not None else 0.0
            P = P @ Wt
            a = a @ Wt + bias
            v = v @ Wt + bias

        elif isinstance(layer, nn.ReLU):
            mask = (v > 0).to(v.dtype).detach()                # gating held fixed
            P = P * mask.unsqueeze(1)
            a = a * mask
            v = torch.relu(v)

        elif isinstance(layer, nn.Conv2d):
            args = (layer.stride, layer.padding, layer.dilation, layer.groups)
            Pf = P.reshape(B * d, *P.shape[2:])
            Pf = F.conv2d(Pf, layer.weight, None, *args)
            P = Pf.reshape(B, d, *Pf.shape[1:])
            a = F.conv2d(a, layer.weight, layer.bias, *args)
            v = F.conv2d(v, layer.weight, layer.bias, *args)

        elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            Pf = layer(P.reshape(B * d, *P.shape[2:]))
            P = Pf.reshape(B, d, *Pf.shape[1:])
            a = layer(a)
            v = layer(v)

        elif isinstance(layer, nn.MaxPool2d):
            k, s, p = layer.kernel_size, layer.stride, layer.padding
            C, Hin, Win = v.shape[1:]
            v, idx = F.max_pool2d(v, k, s, p, return_indices=True)   # idx: (B,C,H',W')
            flat = idx.reshape(B, C, -1)
            a = a.reshape(B, C, Hin * Win).gather(2, flat).reshape(B, C, *v.shape[2:])
            idxP = idx.reshape(B, 1, C, -1).expand(B, d, C, -1)
            P = P.reshape(B, d, C, Hin * Win).gather(3, idxP).reshape(B, d, C, *v.shape[2:])

        else:
            raise TypeError(f"knowledge_matrix: unsupported layer {type(layer).__name__}")

    M = torch.cat((P.transpose(1, 2), a.unsqueeze(-1)), dim=-1)      # (B, K, d + 1)
    return M, v


def grad_knowledge_matrix(model: NN, x: torch.Tensor):
    """
        Differentiable knowledge matrix via the *gradient x input* method
        (the approach of the repo's `gradient-km` branch / GradientMatrixComputer),
        batched over the minibatch with torch.func.

        For a piecewise-linear network f is locally affine, f(x) = J(x) x + c(x),
        so the input columns of M are  J(x) (.) x  (the per-class input Jacobian
        scaled by the input) and the bias column is  c(x) = f(x) - J(x) x.  The
        Jacobian is obtained with `jacrev` (reverse-mode autograd), which costs
        ~K backward passes rather than one forward per input coordinate; it stays
        differentiable w.r.t. the weights, so it can drive a training loss.

        Args:
            model: an NN whose forward is piecewise-linear (ReLU-family + affine
                   layers + max/avg pooling).
            x:     inputs of shape (B, C0, H0, W0).
        Returns:
            M: knowledge matrices, shape (B, K, d + 1).
            y: network outputs,     shape (B, K).   (y == M.sum(-1))
    """
    B = x.shape[0]
    ishape = tuple(x.shape[1:])
    d = math.prod(ishape)
    xf = x.reshape(B, d)

    def single(v):                                   # v: (d,)  ->  (K,)
        return model.forward(v.reshape((1,) + ishape)).flatten()

    J = vmap(jacrev(single))(xf)                     # (B, K, d)
    y = model.forward(x)                             # (B, K)
    grad_x_input = J * xf.unsqueeze(1)               # (B, K, d)
    bias = y - grad_x_input.sum(-1)                  # (B, K)  == FullGrad bias column
    M = torch.cat((grad_x_input, bias.unsqueeze(-1)), dim=-1)
    return M, y


def validate_against_library(model: NN, device="cpu"):
    """Compare the fast differentiable M against `KnowledgeMatrixComputer`."""
    model.eval()
    d = 1
    for s in model.input_shape:
        d *= s
    # 4D input (leading batch dim) so Flatten(start_dim=1) treats channels correctly.
    x = torch.randn((1,) + tuple(model.input_shape),
                    dtype=torch.get_default_dtype(), device=device)
    computer = KnowledgeMatrixComputer(model, batch_size=max(1, d // 4), device=device)
    lib_M = computer.forward(x)                                     # (K, d + 1)
    grad_M, y = grad_knowledge_matrix(model, x)                    # (1, K, d + 1)
    prop_M, _ = knowledge_matrix(model, x)                        # hand-rolled cross-check
    lib_norm = torch.linalg.norm(lib_M).item() + 1e-12
    diff = torch.linalg.norm(lib_M - grad_M[0]).item()             # gradient method vs library
    rel = diff / lib_norm
    prop_rel = torch.linalg.norm(lib_M - prop_M[0]).item() / lib_norm
    sum_err = torch.linalg.norm(y[0] - grad_M[0].sum(-1)).item()   # defining identity
    return diff, rel, sum_err, prop_rel


# --------------------------------------------------------------------------- #
#  Losses on the knowledge matrix
# --------------------------------------------------------------------------- #
def loss_km_eii(M, y):
    """||M(x) - E_ii||_F^2 : pin the whole matrix to the sparse unit E_ii."""
    target = torch.zeros_like(M)
    b = torch.arange(M.shape[0])
    target[b, y, y] = 1.0
    return ((M - target) ** 2).sum(dim=(1, 2)).mean()


def loss_km_offclass(M, y):
    """(sum_c M[i,c] - 1)^2 + sum_{j!=i} ||M[j,:]||^2 : wrong rows -> 0, true logit -> 1."""
    B, K, C = M.shape
    b = torch.arange(B)
    row_sum = M.sum(-1)
    term_correct = (row_sum[b, y] - 1.0) ** 2
    row_sq = (M ** 2).sum(-1)
    off_mask = torch.ones(B, K, device=M.device)
    off_mask[b, y] = 0.0
    term_wrong = (row_sq * off_mask).sum(-1)
    return (term_correct + term_wrong).mean()


def loss_km_rownorm_ce(M, y):
    """Cross-entropy on the per-class row norms of M(x)."""
    return F.cross_entropy(torch.linalg.norm(M, dim=-1), y)


def loss_km_rownorm_margin(M, y):
    """Multiclass hinge on the per-class row norms of M(x)."""
    margin = 1.0
    B = M.shape[0]
    b = torch.arange(B)
    r = torch.linalg.norm(M, dim=-1)
    r_true = r[b, y]
    r_other = r.clone()
    r_other[b, y] = float("-inf")
    return torch.relu(margin + r_other.max(dim=-1).values - r_true).mean()


KM_LOSSES = {
    "km_eii": loss_km_eii,
    "km_offclass": loss_km_offclass,
    "km_rownorm_ce": loss_km_rownorm_ce,
    "km_rownorm_margin": loss_km_rownorm_margin,
}


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def load_mnist1d(path, as_image=True):
    """
        MNIST-1D (Greydanus, github.com/greydanus/mnist1d): 40-dim, 10 classes.
        Inputs are standardized with the training statistics and (by default)
        reshaped to (n, 1, 1, 40) so 2D conv/pool layers apply.
    """
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    dt = torch.get_default_dtype()
    x_tr = torch.tensor(data["x"], dtype=dt)
    y_tr = torch.tensor(data["y"], dtype=torch.long)
    x_te = torch.tensor(data["x_test"], dtype=dt)
    y_te = torch.tensor(data["y_test"], dtype=torch.long)
    mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True) + 1e-8
    x_tr, x_te = (x_tr - mean) / std, (x_te - mean) / std
    if as_image:
        x_tr = x_tr.reshape(-1, 1, 1, 40)
        x_te = x_te.reshape(-1, 1, 1, 40)
    return x_tr, y_tr, x_te, y_te


# --------------------------------------------------------------------------- #
#  Model builders (2D layers over the (1, 1, 40) signal)
# --------------------------------------------------------------------------- #
class MLP(NN):
    def __init__(self, input_shape, num_classes, hidden=64, depth=2):
        super().__init__(input_shape, save=False)
        self.flatten()
        prev = self.get_input_size()
        for _ in range(depth):
            self.linear(in_features=prev, out_features=hidden)
            self.relu()
            prev = hidden
        self.linear(in_features=prev, out_features=num_classes)


class CNN(NN):
    def __init__(self, input_shape, num_classes, channels=(16, 32),
                 kernel=5, hidden=64, pool_out=3):
        super().__init__(input_shape, save=False)
        in_c = input_shape[0]
        pad = kernel // 2
        for out_c in channels:
            self.conv(in_c, out_c, kernel_size=(1, kernel),
                      stride=(1, 1), padding=(0, pad))
            self.relu()
            self.maxpool(kernel_size=(1, 2))
            in_c = out_c
        self.adaptiveavgpool(output_size=(1, pool_out))
        self.flatten()
        self.linear(in_features=channels[-1] * pool_out, out_features=hidden)
        self.relu()
        self.linear(in_features=hidden, out_features=num_classes)


# --------------------------------------------------------------------------- #
#  Training / evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def accuracy(model, x, y, batch=1024):
    model.eval()
    correct = 0
    for s in range(0, x.shape[0], batch):
        out = model.forward(x[s:s + batch])
        correct += (out.argmax(1) == y[s:s + batch]).sum().item()
    return correct / x.shape[0]


SCHEDULERS = ("none", "cosine", "step", "exp", "onecycle", "plateau")


def _make_scheduler(opt, name, epochs, lr):
    """Returns (scheduler_or_None, needs_val_metric)."""
    L = torch.optim.lr_scheduler
    if name in (None, "none", "constant"):
        return None, False
    if name == "cosine":
        return L.CosineAnnealingLR(opt, T_max=epochs), False
    if name == "step":
        return L.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.1), False
    if name == "exp":
        return L.ExponentialLR(opt, gamma=0.01 ** (1.0 / max(1, epochs))), False
    if name == "onecycle":
        return L.OneCycleLR(opt, max_lr=lr, total_steps=epochs), False
    if name == "plateau":
        return L.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5), True
    raise ValueError(f"unknown scheduler {name!r}")


def train(model, x_tr, y_tr, *, mode, epochs, lr, batch_size,
          weight_decay=0.0, km_batch=None, seed=0,
          optimizer="adam", momentum=0.9, scheduler=None,
          eval_data=None, eval_every=1, return_metrics=False):
    """
        mode = 'vanilla' (cross-entropy) or a key of KM_LOSSES.
        km_batch: optional smaller batch size for the (heavier) KM forward.
        optimizer: 'adam' or 'sgd' (SGD uses `momentum`, Nesterov when > 0).
        scheduler: one of SCHEDULERS (none/cosine/step/exp/onecycle/plateau).
        weight_decay: L2 regularization strength (searched in the reg. HPO).
        eval_data: optional (x_val, y_val, x_test, y_test); when given, val/test
            accuracy is measured every `eval_every` epochs and the best-by-val
            checkpoint ("mid-training best") is tracked.
        return_metrics: if True, return a dict with `final` and `best` metrics
            and the per-eval `history`; otherwise return the model.
    """
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              nesterov=momentum > 0, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    sched, needs_metric = _make_scheduler(opt, scheduler, epochs, lr)

    ce = nn.CrossEntropyLoss()
    km_loss = KM_LOSSES.get(mode)
    bs = km_batch if (km_loss is not None and km_batch) else batch_size
    n = x_tr.shape[0]
    g = torch.Generator().manual_seed(seed)

    history = []
    best = {"epoch": 0, "val": -1.0, "test": 0.0, "train": 0.0}
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad()
            if km_loss is not None:
                M, _ = grad_knowledge_matrix(model, xb)   # gradient (jacrev) backend
                loss = km_loss(M, yb)
            else:
                loss = ce(model.forward(xb), yb)
            loss.backward()
            opt.step()

        val_acc = None
        if eval_data is not None and (epoch % eval_every == 0 or epoch == epochs):
            xval, yval, xte, yte = eval_data
            train_acc = accuracy(model, x_tr, y_tr)
            val_acc = accuracy(model, xval, yval)
            test_acc = accuracy(model, xte, yte)
            history.append((epoch, train_acc, val_acc, test_acc))
            if val_acc > best["val"]:
                best = {"epoch": epoch, "val": val_acc, "test": test_acc, "train": train_acc}

        if sched is not None:
            if needs_metric:
                sched.step(val_acc if val_acc is not None else 0.0)
            else:
                sched.step()

    if not return_metrics:
        return model

    final = {"train": accuracy(model, x_tr, y_tr)}
    if eval_data is not None:
        xval, yval, xte, yte = eval_data
        final["val"] = accuracy(model, xval, yval)
        final["test"] = accuracy(model, xte, yte)
    return {"model": model, "final": final, "best": best, "history": history}
