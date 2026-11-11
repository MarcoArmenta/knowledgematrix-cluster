import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Union

from knowledgematrix.neural_net import (
    NN,
    ACTIVATION_LAYERS,
    LINEARIZABLE_LAYERS,
    RMSNorm,
    GLU,
    _ParallelState,
)
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer, _gated_product


class KnowledgeRowComputer:
    """
        Computes selected ROWS of the knowledge matrix by reverse mode
        (one vector-Jacobian product per row), instead of all columns by
        forward mode as KnowledgeMatrixComputer does.

        The linearized network applied to the input columns is a linear
        map, so each of its rows is the gradient of one output coordinate
        of that map -- obtained with a single backward pass through the
        (frozen) linear graph. Cost per requested row is about one forward
        pass, independent of the input size. This is the right direction
        when one needs a few output rows (e.g. the logits of an answer
        token) over many input coordinates, as in interpretability studies
        of language models; extracting a full matrix row-by-row is instead
        much slower than KnowledgeMatrixComputer.

        The rows agree exactly with the corresponding rows of
        KnowledgeMatrixComputer under the same mixer_mode.

        By default the bias entry of a row is the remainder
        `output - A.sum()`, so the row sums to the output by construction
        and that sum checks nothing. `forward(..., independent_bias=True)`
        returns instead the bias computed on its own, by one affine pass
        from the zero input (`KnowledgeMatrixComputer.bias_column`): the
        row then sums to the output only if the linearization is right,
        and `row_outputs` holds the outputs to compare with.

        Scope: transformer-style models -- Embedding start layer, Linear,
        LayerNorm, RMSNorm, Identity, Dropout, Flatten, activation layers,
        MultiHeadAttention (ratio), GatedAttention / GatedDeltaNet / SwiGLU
        (ratio or frozen), GLU (its own alpha split in both modes), and
        residual connections. Convolutional and pooling layers are not
        supported (use KnowledgeMatrixComputer).

        Args:
            model (NN): The neural network.
            mixer_mode (str): "frozen" (default) or "ratio" -- same meaning
                as in KnowledgeMatrixComputer. "frozen" lets attribution
                flow across token positions and is the recommended mode for
                row-level interpretability.
            device (Union[str, None]): Device for the computation. If None,
                the device of the model is used.
    """

    def __init__(
            self,
            model: NN,
            mixer_mode: str = "frozen",
            device: Union[str, None] = None
        ) -> None:
        if mixer_mode not in ("ratio", "frozen"):
            raise ValueError(f'mixer_mode must be "ratio" or "frozen", got {mixer_mode!r}.')
        self.model = model
        self.mixer_mode = mixer_mode
        self.device = device if device is not None else model.device
        # Outputs of the network at the rows of the last `forward` call,
        # shape (len(rows),), on `self.device`.
        self.row_outputs: Union[torch.Tensor, None] = None

    def forward(
            self,
            x: torch.Tensor,
            rows: Sequence[int],
            extract_weff: bool = False,
            independent_bias: bool = False
        ) -> torch.Tensor:
        """
            Computes the requested rows of the knowledge matrix of the
            model at input x.

            Args:
                x (torch.Tensor): The input to the NN (e.g. token ids of
                    shape (1, 1, seq_len) for a language model).
                rows (Sequence[int]): Indices into the FLATTENED output.
                    For a language model with output (1, 1, T, V) the row
                    of token v at position t is t * V + v.
                extract_weff (bool): If True, return the rows of W_eff
                    (the linearization slopes) of shape (len(rows),
                    input_size). If False (default), return the rows of
                    the knowledge matrix A with the bias entry appended,
                    of shape (len(rows), input_size + 1), so that each
                    row sums to the corresponding output coordinate.
                independent_bias (bool): How the bias entry is obtained.
                    False (default): as the remainder
                    `output[row] - A[row].sum()` -- every row then sums to
                    the output by construction, whatever A is.
                    True: independently of the output, by one extra affine
                    pass from the zero input through the linearized network
                    (the bias column of KnowledgeMatrixComputer, same
                    mixer_mode; no column is computed, so the cost is about
                    one forward pass whatever the model size). The entry is
                    then the actual bias of the row, and
                    `rows.sum(1) == row_outputs` is a check that fails when
                    the linearization is wrong. Ignored with extract_weff.
                    The identity holds to the rounding of the terms added,
                    so read the gap relative to `rows.abs().sum(1)`
                    (measured: 2e-16 in float64, 5e-8 to 4e-6 in float32,
                    against 1e-3 for a 1 % error in the row pass; half
                    precision, at 3e-3 to 7e-2, is too coarse for it).
            Returns:
                torch.Tensor: The requested rows. The network outputs at
                    those rows, from the same reference pass, are left in
                    `self.row_outputs` (shape (len(rows),)).
        """
        model = self.model
        with torch.no_grad():
            model.save = True
            out0 = model.forward(x)
            model.save = False
            if model.layer_devices is None:
                model.to(self.device)
            # (a sharded model stays where `NN.shard` put it: the saved
            # states sit on their layer's device and `_linear_forward`
            # follows the same device map)

            start_layer = model._get_start_layer()
            e0 = x
            for i, layer in enumerate(model.layers[:start_layer]):
                e0 = layer(model.to_layer_device(e0, i))
            if e0.dim() == 3:
                e0 = e0.unsqueeze(0)
            e0 = e0.to(self.device)

        e = e0.detach().clone().requires_grad_(True)
        y = self._linear_forward(e).reshape(-1)

        weff_rows = []
        for j in rows:
            grad, = torch.autograd.grad(y[j], e, retain_graph=True)
            weff_rows.append(grad.reshape(-1))
        W = torch.stack(weff_rows)  # (len(rows), input_size)

        # (only the requested entries leave the device of the last layer,
        # not the whole output)
        row_idx = torch.as_tensor(list(rows), device=self.device)
        self.row_outputs = out0.reshape(-1)[row_idx.to(out0.device)].to(self.device)

        if extract_weff:
            return W

        A = W * e0.reshape(1, -1)
        if independent_bias:
            bias = self._independent_bias(out0, row_idx)
        else:
            bias = self.row_outputs - A.sum(dim=1)
        return torch.cat((A, bias.unsqueeze(1)), dim=1)

    def _independent_bias(self, out0: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
        """
            The bias entries of the requested rows, computed without the
            network output: the bias pass of KnowledgeMatrixComputer (one
            affine pass from the zero input, through its `_wire` /
            `_affine_step`) under this computer's mixer_mode, on the states
            the reference pass of `forward` has just saved. Its per-layer
            dispatch is separate from `_linear_forward`; what the two share
            is what they both linearize: the saved states, the weights, and
            the layers' own `frozen_forward` (called here with the biases).
        """
        model = self.model
        computer = KnowledgeMatrixComputer(model, device=self.device, mixer_mode=self.mixer_mode)
        with torch.no_grad():
            # (NN.forward recorded the shape of the embedded input)
            bias = computer._bias_pass((1, *model.input_shape), out0.dtype).reshape(-1)
            return bias[row_idx.to(bias.device)].to(self.device)

    def _linear_forward(self, e: torch.Tensor) -> torch.Tensor:
        """
            The linearized network as a differentiable linear function of
            the embedded input e, replicating exactly the per-layer maps
            KnowledgeMatrixComputer applies to its input columns.
        """
        model = self.model
        start_layer = model._get_start_layer()
        inputs_residuals = [None] * model.get_num_layers()
        branch_snapshots = [None] * model.get_num_layers()
        pstate = _ParallelState()

        B = e
        for i, layer in enumerate(model.layers[start_layer:], start=start_layer):
            B = model.to_layer_device(B, i)   # no-op unless the model is sharded
            _zero = torch.tensor(0.0, device=B.device, dtype=e.dtype)
            if model.parallel_blocks:
                B = model.parallel_step(i, B, pstate)
            if i in model.residuals_starts or i in model.concat_skips_starts:
                inputs_residuals[i] = B
            if i in model.branch_inputs:
                B = model.to_layer_device(branch_snapshots[model.branch_inputs[i]], i)
            if i in model.residuals:
                B = model.apply_residual(B, inputs_residuals, layer=i, affine=False)
                if model.chained_residuals and i in model.residuals_starts:
                    inputs_residuals[i] = B  # the next block skips from the merged stream
            if i in model.concat_skips:
                B = model.apply_concat(B, inputs_residuals, layer=i)
            if i in model.branch_inputs_starts:
                branch_snapshots[i] = B

            if self.mixer_mode == "frozen" and isinstance(layer, LINEARIZABLE_LAYERS):
                B = layer.frozen_forward(B, model.pre_acts[i], affine=False)
            elif isinstance(layer, ACTIVATION_LAYERS):
                vertices = model.acts[i] / model.pre_acts[i]
                vertices = torch.where(
                    torch.isnan(vertices) | torch.isinf(vertices),
                    _zero,
                    vertices
                )
                B = B * vertices
            elif isinstance(layer, GLU):
                # Same linear map as the column pass of
                # KnowledgeMatrixComputer: the constant act(0) of the gate
                # carries no input attribution and ends up in the bias entry.
                u, g, v = model.gated_products[i]
                c = layer.act(torch.zeros((), dtype=u.dtype, device=u.device))
                ratio = (g - c) / u
                ratio = torch.where(torch.isnan(ratio) | torch.isinf(ratio), _zero, ratio)
                M_g = F.linear(B, layer.gate_proj.weight) * ratio
                M_v = F.linear(B, layer.value_proj.weight)
                B = _gated_product(M_g, g, M_v, v, layer.alpha)
            elif isinstance(layer, nn.Linear):
                B = F.linear(B, layer.weight)
            elif isinstance(layer, nn.LayerNorm):
                B = B * layer.weight / torch.sqrt(model.layernorms[i][1] + layer.eps)
            elif isinstance(layer, RMSNorm):
                B = B * layer.weight / model.layernorms[i]
            elif isinstance(layer, nn.Flatten):
                B = layer(B)
            elif isinstance(layer, (nn.Identity, nn.Dropout)):
                pass
            else:
                raise NotImplementedError(
                    f"KnowledgeRowComputer does not support layer type "
                    f"{type(layer).__name__} (layer {i}); use "
                    f"KnowledgeMatrixComputer for this model."
                )
        return B
