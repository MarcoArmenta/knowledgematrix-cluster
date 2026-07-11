import torch
from torch import nn
from typing import Union
from torch.nn import functional as F

from knowledgematrix.neural_net import NN, MultiHeadAttention


def compose(m2: torch.Tensor, m1: torch.Tensor) -> torch.Tensor:
    """Augmented product of two knowledge matrices: the contribution-form KM of
    the composed segment, exact because both factors are frozen by the same
    reference pass.

    Knowledge matrices are contribution-form (columns are per-input-dimension
    contributions, so `M·1 = f(x)`), i.e. `m = A·diag(cut_in) | b`, where the
    linear columns carry the input activations at the segment's input cut. The
    shared-cut activations between the two segments are recovered exactly as the
    row-sums of `m1` (`m1·1 = cut`), which de-scale `m2`'s columns back to the
    plain affine operator before the product. The result is again contribution
    form w.r.t. `m1`'s input:
        [A2·A1·diag(cut_in) | A2·b1 + b2].
    """
    cut = m1.sum(1)                             # activations at the shared cut
    scale = m2[:, :-1] / cut.unsqueeze(0)       # de-scale m2's columns to the affine operator
    scale = torch.where(
        torch.isnan(scale) | torch.isinf(scale),
        torch.zeros_like(scale),
        scale
    )
    lin = scale @ m1[:, :-1]
    bias = scale @ m1[:, -1] + m2[:, -1]
    return torch.cat((lin, bias.unsqueeze(-1)), dim=-1)


class KnowledgeMatrixComputer:
    """
        A class to compute the knowledge matrix of a neural network.

        Args:
            model (NN): The neural network to compute the knowledge matrix of.
            batch_size (int): The batch size to use when computing the knowledge matrix.
            device (Union[str, None]): The device to use when computing the knowledge matrix. If None, the device of the model is used.
    """

    def __init__(
            self,
            model: NN,
            batch_size: int = 1,
            device: Union[str, None] = None,
            attention_mode: str = "monolithic"
        ) -> None:
        if attention_mode not in ("monolithic", "frozen_pattern"):
            raise ValueError(f"attention_mode must be 'monolithic' or 'frozen_pattern', got {attention_mode!r}")
        self.attention_mode = attention_mode
        self.model = model
        self.batch_size = batch_size
        self.layers = model.layers
        self.device = device if device is not None else model.device
        self.in_c, self.in_h, self.in_w = model.input_shape
        self.input_size = self.in_c*self.in_h*self.in_w

        # Saves the output of the NN on the current sample in the forward method
        self.current_output: Union[NN, None] = None

    def _linear_step(self, B: torch.Tensor, i: int, layer) -> torch.Tensor:
        """Probe-pass (weights-only) transform of layer i. Unknown layers
        (Dropout in eval, etc.) pass through unchanged."""
        if isinstance(layer, MultiHeadAttention) and self.attention_mode == "frozen_pattern":
            pattern = layer.attn_pattern           # (1, 1, H, T, T) from reference pass
            batch, C, T, D = B.shape
            v = (layer.V.weight @ B.transpose(-1, -2)).transpose(-1, -2)
            v = v.view(batch, C, T, layer.num_heads, layer.d_head).transpose(2, 3)
            out = pattern @ v                      # broadcasts over probe batch
            out = out.transpose(2, 3).contiguous().view(batch, C, T, D)
            B = (layer.O.weight @ out.transpose(-1, -2)).transpose(-1, -2)
        elif isinstance(layer, (nn.ELU, nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU, nn.Mish, nn.Softmax, MultiHeadAttention)):
            pre_act = self.model.pre_acts[i]
            post_act = self.model.acts[i]
            vertices = post_act / pre_act
            vertices = torch.where(
                torch.isnan(vertices) | torch.isinf(vertices),
                self._zero,
                vertices
            ).squeeze(0)
            B = B * vertices
        elif isinstance(layer, nn.Conv2d):
            B = F.conv2d(B, layer.weight, None, stride=layer.stride, padding=layer.padding)
        elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Flatten)):
            B = layer(B)
        elif isinstance(layer, nn.Linear):
            B = (layer.weight @ B.transpose(-1, -2)).transpose(-1, -2)
        elif isinstance(layer, nn.BatchNorm2d):
            B = B * (layer.weight / torch.sqrt(layer.running_var + layer.eps)).view(1, -1, 1, 1)
        elif isinstance(layer, nn.LayerNorm):
            B = B * layer.weight / torch.sqrt(self.model.layernorms[i][1] + layer.eps)
        elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
            pool = self.model.maxpool_indices[i]
            batch_indices = torch.arange(B.shape[0], device=self.device).view(-1, 1, 1, 1)
            channel_indices = torch.arange(pool.shape[1], device=self.device).view(1, -1, 1, 1)
            row_indices = pool // B.shape[2] if self.IN_2D else pool
            col_indices = pool % B.shape[3]
            B = B[batch_indices, channel_indices, row_indices, col_indices]
        return B

    def _affine_step(self, a: torch.Tensor, i: int, layer) -> torch.Tensor:
        """Bias-pass (full affine) transform of layer i."""
        if isinstance(layer, MultiHeadAttention) and self.attention_mode == "frozen_pattern":
            pattern = layer.attn_pattern
            batch, C, T, D = a.shape
            v = layer.V(a)                         # with bias
            v = v.view(batch, C, T, layer.num_heads, layer.d_head).transpose(2, 3)
            out = pattern @ v
            out = out.transpose(2, 3).contiguous().view(batch, C, T, D)
            a = layer.O(out)                       # with bias
        elif isinstance(layer, (nn.ELU, nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU, nn.Mish, nn.Softmax, MultiHeadAttention)):
            pre_act = self.model.pre_acts[i]
            post_act = self.model.acts[i]
            vertices = post_act / pre_act
            vertices = torch.where(
                torch.isnan(vertices) | torch.isinf(vertices),
                self._zero,
                vertices
            )
            a = a * vertices
        elif isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.BatchNorm2d, nn.Flatten, nn.Linear)):
            a = layer(a)
        elif isinstance(layer, nn.LayerNorm):
            a = ((a - self.model.layernorms[i][0]) / torch.sqrt(self.model.layernorms[i][1] + layer.eps)) * layer.weight + layer.bias
        elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
            pool = self.model.maxpool_indices[i]
            batch_indices = torch.arange(pool.shape[0], device=self.device).view(-1, 1, 1, 1)
            channel_indices = torch.arange(pool.shape[1], device=self.device).view(1, -1, 1, 1)
            row_indices = pool // a.shape[2] if self.IN_2D else pool
            col_indices = pool % a.shape[3]
            a = a[batch_indices, channel_indices, row_indices, col_indices]
        return a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Computes the knowledge matrix of a NN at a given input point.
            Args:
                x (torch.Tensor): The input to the NN
            Returns:
                torch.Tensor: The knowledge matrix of the NN at the input point
        """
        with torch.no_grad():
            # Saves activations and pre-activations
            self.model.save = True
            self.current_output = self.model.forward(x)
            self.model.save = False
            self.model.to(self.device)
            start_layer = self.model._get_start_layer()
            for layer in self.model.layers[:start_layer]:
                x = layer(x)
            if start_layer > 0:
                C, H, W = x.shape[1], x.shape[2], x.shape[3]
            else:
                C, H, W = x.shape[0], x.shape[1], x.shape[2]

            # Total number of positions and batches needed
            total_positions = C*H*W
            num_batches = (total_positions + self.batch_size - 1)//self.batch_size
            output_size = self.current_output.numel()
            dtype = self.current_output.dtype

            self.IN_2D = (W > 1)  # Wether the input is of shape (C,H,W) or (C,L,1)

            x = x.to(self.device)
            self._zero = torch.tensor(0.0, device=self.device, dtype=dtype)
            inputs_residuals = [None] * self.model.get_num_layers()
            A = torch.empty((output_size, total_positions), device=self.device, dtype=dtype)

            for batch in range(num_batches):
                # Compute batch indices
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, total_positions)
                current_batch_size = end - start

                # Create indices for this batch
                indices = torch.arange(start, end, device=self.device)
                c = indices // (H*W)
                remaining = indices % (H*W)
                h = remaining // W
                w = remaining % W

                # Create batched input for this chunk
                batched_input = torch.zeros((current_batch_size,C,H,W), device=self.device, dtype=dtype)
                batched_input[torch.arange(current_batch_size, device=self.device),c,h,w] = x.flatten()[start:end]

                B = batched_input
                for i, layer in enumerate(self.layers[start_layer:], start=start_layer):
                    if i in self.model.residuals:
                        B = self.model.apply_residual(B, inputs_residuals, layer=i, affine=False)
                    if i in self.model.residuals_starts:
                        inputs_residuals[i] = B
                    B = self._linear_step(B, i, layer)

                B = B.reshape(-1, output_size)
                A[:, start:end] = B.T

            # Process bias and batch norm terms by iterating through layers again
            # Computing activation ratios and applying appropriate transformations
            if self.model._has_bias() or self.model._has_batchnorm() or self.model._has_layernorm() or len(self.model.residuals) > 0:
                a = torch.zeros(x.shape, device=self.device, dtype=dtype)
                if len(x.shape) == 3:
                    a = a.unsqueeze(0)
                for i, layer in enumerate(self.layers[start_layer:], start=start_layer):
                    if i in self.model.residuals:
                        a = self.model.apply_residual(a, inputs_residuals, layer=i)
                    if i in self.model.residuals_starts:
                        inputs_residuals[i] = a
                    a = self._affine_step(a, i, layer)

                a = a.reshape(-1, output_size)
                return torch.cat((A, a.T), dim=-1)

            return A

    def segment(self, x: torch.Tensor, start_cut: int, end_cut: int,
                final_position_only: bool = False) -> torch.Tensor:
        """KM of layers [start_cut, end_cut) at input x, under the cut-point
        convention: the residual application AT start_cut belongs upstream (skipped);
        the residual application AT end_cut belongs to this segment (trailing step).
        Cut values are residual-stream tensors saved by the reference pass."""
        n = self.model.get_num_layers()
        if not (self.model._get_start_layer() <= start_cut < end_cut <= n):
            raise ValueError(f"invalid cuts ({start_cut}, {end_cut})")
        if final_position_only and end_cut != n:
            raise ValueError("final_position_only requires end_cut == num_layers")

        with torch.no_grad():
            # Reference pass: activations, patterns, LN stats, stream values
            self.model.save = True
            self.current_output = self.model.forward(x)
            self.model.save = False
            self.model.to(self.device)

            x0 = self.model.stream[start_cut].to(self.device)      # (1, C, T, D)
            seg_out = self.current_output if end_cut == n else self.model.stream[end_cut]
            dtype = self.current_output.dtype
            self._zero = torch.tensor(0.0, device=self.device, dtype=dtype)
            _, C, H, W = x0.shape
            self.IN_2D = (W > 1)
            total_positions = C * H * W
            out_numel = seg_out.numel()
            if final_position_only:
                seq_len = self.current_output.shape[-2]
                vocab = self.current_output.shape[-1]
                row_lo, row_hi = (seq_len - 1) * vocab, seq_len * vocab
            else:
                row_lo, row_hi = 0, out_numel
            num_batches = (total_positions + self.batch_size - 1) // self.batch_size
            A = torch.empty((row_hi - row_lo, total_positions), device=self.device, dtype=dtype)

            def run_probe(P, affine):
                """Push P through the segment. affine=False: weights-only probe pass;
                affine=True: full-affine bias pass."""
                inputs_residuals = [None] * n
                step = self._affine_step if affine else self._linear_step
                for i, layer in enumerate(self.layers[start_cut:end_cut], start=start_cut):
                    if i in self.model.residuals and i > start_cut:
                        P = self.model.apply_residual(P, inputs_residuals, layer=i, affine=affine)
                    if i in self.model.residuals_starts:
                        inputs_residuals[i] = P
                    P = step(P, i, layer)
                if end_cut < n and end_cut in self.model.residuals:
                    P = self.model.apply_residual(P, inputs_residuals, layer=end_cut, affine=affine)
                return P

            for batch in range(num_batches):
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, total_positions)
                cbs = end - start
                indices = torch.arange(start, end, device=self.device)
                c = indices // (H * W)
                remaining = indices % (H * W)
                h = remaining // W
                w = remaining % W
                B = torch.zeros((cbs, C, H, W), device=self.device, dtype=dtype)
                B[torch.arange(cbs, device=self.device), c, h, w] = x0.flatten()[start:end]
                B = run_probe(B, affine=False)
                A[:, start:end] = B.reshape(cbs, -1)[:, row_lo:row_hi].T

            a = torch.zeros(x0.shape, device=self.device, dtype=dtype)
            a = run_probe(a, affine=True)
            a = a.reshape(1, -1)[:, row_lo:row_hi]
            return torch.cat((A, a.T), dim=-1)
