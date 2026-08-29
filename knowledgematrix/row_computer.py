import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Union

from knowledgematrix.neural_net import (
    NN,
    ACTIVATION_LAYERS,
    LINEARIZABLE_LAYERS,
    RMSNorm,
)


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

        Scope: transformer-style models -- Embedding start layer, Linear,
        LayerNorm, RMSNorm, Identity, Dropout, Flatten, activation layers,
        MultiHeadAttention (ratio), GatedAttention / GatedDeltaNet / SwiGLU
        (ratio or frozen), and residual connections. Convolutional and
        pooling layers are not supported (use KnowledgeMatrixComputer).

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

    def forward(
            self,
            x: torch.Tensor,
            rows: Sequence[int],
            extract_weff: bool = False
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
            Returns:
                torch.Tensor: The requested rows.
        """
        model = self.model
        with torch.no_grad():
            model.save = True
            out0 = model.forward(x)
            model.save = False
            model.to(self.device)

            start_layer = model._get_start_layer()
            e0 = x
            for layer in model.layers[:start_layer]:
                e0 = layer(e0)
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

        if extract_weff:
            return W

        A = W * e0.reshape(1, -1)
        out_flat = out0.reshape(-1).to(self.device)
        row_idx = torch.as_tensor(list(rows), device=self.device)
        bias = out_flat[row_idx] - A.sum(dim=1)
        return torch.cat((A, bias.unsqueeze(1)), dim=1)

    def _linear_forward(self, e: torch.Tensor) -> torch.Tensor:
        """
            The linearized network as a differentiable linear function of
            the embedded input e, replicating exactly the per-layer maps
            KnowledgeMatrixComputer applies to its input columns.
        """
        model = self.model
        start_layer = model._get_start_layer()
        _zero = torch.tensor(0.0, device=self.device, dtype=e.dtype)
        inputs_residuals = [None] * model.get_num_layers()
        branch_snapshots = [None] * model.get_num_layers()

        B = e
        for i, layer in enumerate(model.layers[start_layer:], start=start_layer):
            if i in model.residuals_starts or i in model.concat_skips_starts:
                inputs_residuals[i] = B
            if i in model.branch_inputs:
                B = branch_snapshots[model.branch_inputs[i]]
            if i in model.residuals:
                B = model.apply_residual(B, inputs_residuals, layer=i, affine=False)
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
