import torch
import warnings
from torch import nn
from typing import Union
from torch.nn import functional as F

from knowledgematrix.neural_net import (
    NN,
    ACTIVATION_LAYERS,
    LINEARIZABLE_LAYERS,
    RMSNorm,
    MultiHeadAttention,
    _ParallelState,
    GLU,
    DEFAULT_GATED_PRODUCT_ALPHA,
)

# DEFAULT_GATED_PRODUCT_ALPHA lives in neural_net.py (single source of truth) and is
# re-exported here for backward compatibility; matrix_computer imports from neural_net,
# never the reverse (avoids a circular import).

# Distinct alpha values already warned about (once per value, per process). Membership is
# recorded AFTER the warning is issued so a suppressed or raised (-W error) warning does
# not mark the alpha as already-warned; later models/blocks with a different alpha still warn.
_gated_alpha_warned: set = set()


def _gated_product(M_g, g, M_v, v, alpha):
    return alpha * (v * M_g) + (1.0 - alpha) * (g * M_v)


def _warn_gated_product_alpha(alpha):
    if alpha in _gated_alpha_warned:
        return
    warnings.warn(
        f"Knowledge matrix uses a gated-product attribution split with alpha={alpha} "
        f"(0.5=symmetric). The invariant mat.sum(1)==forward holds for any alpha; alpha "
        f"only reweights gate-path vs value-path input attribution, which matters for "
        f"whole-model KM analysis. See the GLU docstring in knowledgematrix/neural_net.py.",
        UserWarning,
    )
    _gated_alpha_warned.add(alpha)


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
            mixer_mode (str): How token-mixing / gated layers (MultiHeadAttention,
                RopeAttention, GatedAttention, GatedDeltaNet, SwiGLU, GeGLU)
                are linearized. "ratio" (default)
                applies the elementwise post/pre activation ratio, like any
                other activation -- attribution then stays within each token
                position (the matrix is block diagonal over positions).
                "frozen" applies each such layer as a linear map with its
                routing (attention weights, recurrence gates, GLU gates)
                frozen at the actual input -- attribution then flows across
                positions through the value paths. Both modes reproduce the
                forward pass exactly (mat.sum(1) == forward).
                A GLU layer (NN.glu) is not affected by this argument: it
                is linearized by its own `alpha` in both modes.
            attention_mode (str): "monolithic" (default) leaves the choice to
                mixer_mode. "frozen_pattern" is mixer_mode="frozen" restricted
                to MultiHeadAttention: those layers are applied as the linear
                value-path map O·(pattern ⊗ V) with the softmax pattern frozen
                at the actual input, while every other layer keeps its
                mixer_mode treatment. It is the same linearization as "frozen"
                (the layer's `frozen_forward`), so for a network whose only
                linearizable layers are MultiHeadAttention (GPT2, GPTNeoX,
                Transformer) the two give the same matrix. mixer_mode="frozen"
                already includes it.

        A network split over several devices with `NN.shard` is computed as
        the pipeline it is: `forward` and `segment` never move it, the probe
        batch and the bias pass go to the device of each layer in turn (the
        saved states of the reference pass already live there), and only the
        finished columns are brought to `device` -- which defaults to
        `model.device`, the first stage -- where the returned matrix lives.
        The matrix equals that of the unsplit network. `current_output` stays
        where the forward pass leaves it (the device of the last layer).
    """

    def __init__(
            self,
            model: NN,
            batch_size:int = 1,
            device:Union[str, None] = None,
            mixer_mode:str = "ratio",
            attention_mode:str = "monolithic"
        ) -> None:
        if mixer_mode not in ("ratio", "frozen"):
            raise ValueError(f'mixer_mode must be "ratio" or "frozen", got {mixer_mode!r}.')
        if attention_mode not in ("monolithic", "frozen_pattern"):
            raise ValueError(f"attention_mode must be 'monolithic' or 'frozen_pattern', got {attention_mode!r}")
        self.model = model
        self.batch_size = batch_size
        self.layers = model.layers
        self.mixer_mode = mixer_mode
        self.attention_mode = attention_mode
        self.device = device if device is not None else model.device
        # input_shape may be 1-D (a bare feature count, e.g. MNIST-1D), 2-D
        # (channels, length -- 1-D convolutional nets) or the 3-D image
        # (channels, height, width). Normalize to (C, H, W) with singleton
        # axes so the batched column construction below is shape-agnostic.
        shape = model.input_shape
        if isinstance(shape, int):
            self.in_c, self.in_h, self.in_w = 1, shape, 1
        elif len(shape) == 1:
            self.in_c, self.in_h, self.in_w = 1, int(shape[0]), 1
        elif len(shape) == 2:
            self.in_c, self.in_w = (int(v) for v in shape)
            self.in_h = 1
        elif len(shape) == 3:
            self.in_c, self.in_h, self.in_w = (int(v) for v in shape)
        else:
            raise ValueError(
                f"input_shape must have 1, 2 or 3 dimensions, got {shape!r}.")
        self.input_size = self.in_c*self.in_h*self.in_w

        # Saves the output of the NN on the current sample in the forward method
        self.current_output: Union[NN, None] = None

    def _frozen_layer_types(self) -> tuple:
        """The layer types applied through their frozen-routing linear map
        (`frozen_forward`) instead of the elementwise post/pre ratio."""
        if self.mixer_mode == "frozen":
            return LINEARIZABLE_LAYERS
        if self.attention_mode == "frozen_pattern":
            return (MultiHeadAttention,)
        return ()

    def _wire(self, P: torch.Tensor, i: int, inputs_residuals: list, branch_snapshots: list,
              pstate: _ParallelState, affine: bool, seg_start: int = -1) -> torch.Tensor:
        """The skip-connection bookkeeping of NN.forward at loop index i (run
        BEFORE layer i), applied to a probe batch (affine=False) or to the bias
        pass (affine=True). Inside segment(), the merges AT the segment's first
        index belong to the upstream segment and are skipped (seg_start)."""
        model = self.model
        # A no-op unless the model is split over devices (NN.shard): the
        # pass then moves to the device of layer i, exactly as NN.forward
        # does, and finds there the states saved by the reference pass.
        P = model.to_layer_device(P, i)
        if model.parallel_blocks:
            P = model.parallel_step(i, P, pstate, seg_start=seg_start)
        if i in model.residuals_starts or i in model.concat_skips_starts:
            inputs_residuals[i] = P
        if i in model.branch_inputs:
            P = model.to_layer_device(branch_snapshots[model.branch_inputs[i]], i)
        if i in model.residuals and i != seg_start:
            P = model.apply_residual(P, inputs_residuals, layer=i, affine=affine)
            if model.chained_residuals and i in model.residuals_starts:
                inputs_residuals[i] = P  # the next block skips from the merged stream
        if i in model.concat_skips:
            P = model.apply_concat(P, inputs_residuals, layer=i)
        if i in model.branch_inputs_starts:
            branch_snapshots[i] = P
        return P

    def _zero_for(self, t: torch.Tensor) -> torch.Tensor:
        """The scalar 0 of the current pass on the device of `t`: `self._zero`
        itself, except on the other stages of a network split over devices."""
        zero = self._zero
        return zero if zero.device == t.device else zero.to(t.device)

    def _layer_device(self, i: int):
        """The device layer i is applied on: its own for a network split
        over devices (which is never moved), the computer's otherwise."""
        if self.model.layer_devices is None:
            return self.device
        return self.model.layer_devices[i]

    def _linear_step(self, B: torch.Tensor, i: int, layer) -> torch.Tensor:
        """Probe-pass (weights-only) transform of layer i. Unknown layers
        (Dropout in eval, Identity, etc.) pass through unchanged."""
        if isinstance(layer, self._frozen_layer_types()):
            B = layer.frozen_forward(B, self.model.pre_acts[i], affine=False)
        elif isinstance(layer, ACTIVATION_LAYERS):
            # Get activation ratios
            pre_act = self.model.pre_acts[i]
            post_act = self.model.acts[i]
            vertices = post_act / pre_act
            vertices = torch.where(
                torch.isnan(vertices) | torch.isinf(vertices),
                self._zero_for(vertices),
                vertices
            ).squeeze(0)  # Remove original batch dim
            B = B * vertices
        elif isinstance(layer, GLU):
            _warn_gated_product_alpha(layer.alpha)
            u, g, v = self.model.gated_products[i]
            u = u.squeeze(0); g = g.squeeze(0); v = v.squeeze(0)   # (hidden,), last axis = features
            # Origin-shift: act(u) = ratio*u + c with c=act(0). This is exact even
            # when u==0 and act(0)!=0 (sigmoid: c=0.5); a plain g/u would nan->0 and
            # silently drop the gate row. c is a constant (no input attribution), so it
            # is NOT added in this weff pass -- it lands in the bias column (bias pass).
            c = layer.act(torch.zeros((), dtype=u.dtype, device=u.device))
            M_u = (layer.gate_proj.weight @ B.transpose(-1, -2)).transpose(-1, -2)   # weight only
            ratio = (g - c) / u
            ratio = torch.where(torch.isnan(ratio) | torch.isinf(ratio), self._zero_for(ratio), ratio)
            M_g = M_u * ratio
            M_v = (layer.value_proj.weight @ B.transpose(-1, -2)).transpose(-1, -2)  # weight only
            B = _gated_product(M_g, g, M_v, v, layer.alpha)
        elif isinstance(layer, nn.Conv2d):
            B = F.conv2d(B, layer.weight, None, stride=layer.stride, padding=layer.padding,
                         dilation=layer.dilation, groups=layer.groups)
        elif isinstance(layer, nn.ConvTranspose2d):
            B = F.conv_transpose2d(B, layer.weight, None, stride=layer.stride, padding=layer.padding,
                                   output_padding=layer.output_padding, dilation=layer.dilation, groups=layer.groups)
        elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Upsample, nn.PixelShuffle)):
            B = layer(B)
        elif isinstance(layer, nn.Linear):
            B = (layer.weight @ B.transpose(-1,-2)).transpose(-1,-2)
        elif isinstance(layer, nn.BatchNorm2d):
            B = B * (layer.weight/torch.sqrt(layer.running_var+layer.eps)).view(1,-1,1,1)
        elif isinstance(layer, nn.LayerNorm):
            B = B * layer.weight/torch.sqrt(self.model.layernorms[i][1]+layer.eps)
        elif isinstance(layer, RMSNorm):
            B = B * layer.weight / self.model.layernorms[i]
        elif isinstance(layer, nn.GroupNorm):
            var_expanded = self.model.layernorms[i][1]
            B = B * (layer.weight.view(1, -1, 1, 1) / torch.sqrt(var_expanded + layer.eps))
        elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
            pool = self.model.maxpool_indices[i]
            batch_indices = torch.arange(B.shape[0], device=B.device).view(-1,1,1,1)
            channel_indices = torch.arange(pool.shape[1], device=B.device).view(1,-1,1,1)
            row_indices = pool // B.shape[2] if self.IN_2D else pool
            col_indices = pool % B.shape[3]
            B = B[batch_indices, channel_indices, row_indices, col_indices]
        return B

    def _affine_step(self, a: torch.Tensor, i: int, layer) -> torch.Tensor:
        """Bias-pass (full affine) transform of layer i."""
        if isinstance(layer, self._frozen_layer_types()):
            a = layer.frozen_forward(a, self.model.pre_acts[i], affine=True)
        elif isinstance(layer, ACTIVATION_LAYERS):
            pre_act = self.model.pre_acts[i]
            post_act = self.model.acts[i]
            vertices = post_act / pre_act
            undefined = torch.isnan(vertices) | torch.isinf(vertices)
            zero = self._zero_for(vertices)
            vertices = torch.where(
                undefined,
                zero,
                vertices
            )
            a = a * vertices
            # Where the ratio does not exist (a pre-activation of
            # exactly 0), what the layer outputs cannot be
            # attributed to that input coordinate: it is a
            # constant of the linearization and belongs to the
            # bias column, or the row would no longer sum to the
            # output. It is 0 for an activation with f(0) = 0,
            # but not for a sigmoid, nor for a mixing layer
            # (attention, gated MLP) fed by a norm whose scale is
            # exactly 0 in some coordinate -- which trained
            # checkpoints do contain (Gemma-3).
            a = a + torch.where(undefined & torch.isfinite(post_act), post_act, zero)
        elif isinstance(layer, GLU):
            u, g, v = self.model.gated_products[i]
            u = u.squeeze(0); g = g.squeeze(0); v = v.squeeze(0)
            # Origin-shift (see weff pass). Here the constant c=act(0) DOES belong:
            # this is the bias/constant column, so M_g = ratio*u_contrib + c makes the
            # gate row-sum ratio*u + c = (g-c)+c = g exactly (sigmoid u==0 included).
            c = layer.act(torch.zeros((), dtype=u.dtype, device=u.device))
            M_u = layer.gate_proj(a)     # full linear: W1·a + b1
            ratio = (g - c) / u
            ratio = torch.where(torch.isnan(ratio) | torch.isinf(ratio), self._zero_for(ratio), ratio)
            M_g = M_u * ratio + c
            M_v = layer.value_proj(a)    # W2·a + b2
            a = _gated_product(M_g, g, M_v, v, layer.alpha)
        elif isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.BatchNorm2d, nn.Flatten, nn.Linear, nn.Upsample, nn.PixelShuffle)):
            a = layer(a)
        elif isinstance(layer, nn.LayerNorm):
            a = ((a - self.model.layernorms[i][0])/torch.sqrt(self.model.layernorms[i][1]+layer.eps))*layer.weight + layer.bias
        elif isinstance(layer, RMSNorm):
            a = a * layer.weight / self.model.layernorms[i]
        elif isinstance(layer, nn.GroupNorm):
            mean_expanded = self.model.layernorms[i][0]
            var_expanded = self.model.layernorms[i][1]
            a = ((a - mean_expanded) / torch.sqrt(var_expanded + layer.eps)) * layer.weight.view(1, -1, 1, 1) + layer.bias.view(1, -1, 1, 1)
        elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
            pool = self.model.maxpool_indices[i]
            batch_indices = torch.arange(pool.shape[0], device=a.device).view(-1,1,1,1)
            channel_indices = torch.arange(pool.shape[1], device=a.device).view(1,-1,1,1)
            row_indices = pool // a.shape[2] if self.IN_2D else pool
            col_indices = pool % a.shape[3]
            a = a[batch_indices, channel_indices, row_indices, col_indices]
        return a

    def forward(self, x: torch.Tensor, extract_weff: bool = False) -> torch.Tensor:
        """
            Computes the knowledge matrix of a NN at a given input point.
            Args:
                x (torch.Tensor): The input to the NN
                extract_weff (bool): If True, return W_eff[j, i] = df[j]/dx_i
                    (the local linearization slope) of shape
                    (output_size, input_size). If False (default), return the
                    standard knowledge matrix A with A[j, i] = W_eff[j, i] * x_i
                    and an appended bias column of shape
                    (output_size, input_size + 1). W_eff is exact only within
                    the activation region of x (ReLU-family sign patterns and
                    MaxPool argmax indices); re-extract after perturbations
                    that cross region boundaries.
            Returns:
                torch.Tensor: The knowledge matrix A (default) or slope W_eff
                    (extract_weff=True) of the NN at the input point.
        """
        with torch.no_grad():
            # Saves activations and pre-activations
            self.model.save = True
            if self.model.layer_devices is None:
                self.model.to(self.device)
            # (a network split over devices stays where `NN.shard` put it:
            # moving it would collapse the split)
            self.current_output = self.model.forward(x)
            self.model.save = False
            start_layer = self.model._get_start_layer()
            for i, layer in enumerate(self.model.layers[:start_layer]):
                x = layer(self.model.to_layer_device(x, i))
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

            # The probe batch starts on the device of the first layer it
            # goes through (the computer's own device unless the network is
            # split) and follows the layers from there, see `_wire`.
            in_device = self._layer_device(start_layer)
            x = x.to(in_device)
            self._zero = torch.tensor(0.0, device=self.device, dtype=dtype)
            inputs_residuals = [None] * self.model.get_num_layers()
            branch_snapshots = [None] * self.model.get_num_layers()
            A = torch.empty((output_size, total_positions), device=self.device, dtype=dtype)

            for batch in range(num_batches):
                # Compute batch indices
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, total_positions)
                current_batch_size = end - start

                # Create indices for this batch
                indices = torch.arange(start, end, device=in_device)
                c = indices // (H*W)
                remaining = indices % (H*W)
                h = remaining // W
                w = remaining % W

                # Create batched input for this chunk
                batched_input = torch.zeros((current_batch_size,C,H,W), device=in_device, dtype=dtype)
                if extract_weff:
                    batched_input[torch.arange(current_batch_size, device=in_device),c,h,w] = 1.0
                else:
                    batched_input[torch.arange(current_batch_size, device=in_device),c,h,w] = x.flatten()[start:end]

                B = batched_input
                pstate = _ParallelState()
                for i, layer in enumerate(self.layers[start_layer:], start=start_layer):
                    # Skip-connection bookkeeping, then the layer itself
                    # (Conv2d, AvgPool2d, Linear, BatchNorm2d, MaxPool2d, etc.)
                    # applying the appropriate transformations and handling activation ratios
                    B = self._wire(B, i, inputs_residuals, branch_snapshots, pstate, affine=False)
                    B = self._linear_step(B, i, layer)

                B = B.reshape(-1, output_size)
                A[:, start:end] = B.T.to(self.device)   # from the last stage of a split network

            if extract_weff:
                return A

            # Process bias and batch norm terms by iterating through layers again
            # Computing activation ratios and applying appropriate transformations
            # (one condition for this method and NN.get_matrix_shape: any
            # source of a constant term, biases inside composite layers and
            # the mixing layers themselves included)
            if self.model.has_bias_column():
                a = self._bias_pass(x.shape, dtype)
                a = a.reshape(-1, output_size)
                return torch.cat((A, a.T.to(self.device)), dim=-1)

            return A

    def _bias_pass(self, shape, dtype: torch.dtype) -> torch.Tensor:
        """
            The bias pass: the zero input pushed through the full AFFINE
            linearized network (`_wire` + `_affine_step`), which leaves
            exactly the constant term of the linearization -- the bias
            column. One pass, no columns.

            It reads the states saved by a reference pass (`model.save =
            True`) at the input in question, which the caller must have
            run; `shape` is the shape of the input after the start layers
            (embedding, positional encoding) and `dtype` that of the
            output. Runs under the caller's `torch.no_grad()`.

            Returns:
                torch.Tensor: the bias in the shape of the network output,
                    on the device of the last layer.
        """
        model = self.model
        start_layer = model._get_start_layer()
        self._zero = torch.tensor(0.0, device=self.device, dtype=dtype)
        self.IN_2D = (shape[-1] > 1)
        a = torch.zeros(tuple(shape), device=self._layer_device(start_layer), dtype=dtype)
        if len(shape) == 3:
            a = a.unsqueeze(0)
        inputs_residuals = [None] * model.get_num_layers()
        branch_snapshots = [None] * model.get_num_layers()
        pstate = _ParallelState()
        for i, layer in enumerate(self.layers[start_layer:], start=start_layer):
            a = self._wire(a, i, inputs_residuals, branch_snapshots, pstate, affine=True)
            a = self._affine_step(a, i, layer)
        return a

    def bias_column(self, x: torch.Tensor) -> torch.Tensor:
        """
            The bias column of the knowledge matrix at x, alone: one
            reference pass and one affine pass, whatever the input size --
            none of the input columns is computed.

            It is the last column of `forward(x)` whenever that has one
            (same code, same arithmetic), under the same mixer_mode /
            attention_mode, and all zeros for a network whose linearization
            has no constant term. (The pass is run whether or not
            `NN.has_bias_column()` holds, so it also returns the constant
            that `forward` has no column for: an elementwise activation
            with f(0) != 0 at a pre-activation of exactly 0, in a network
            with no bias, norm, residual or mixing layer.)

            Being computed from the zero input, it does not use the network
            output: with the input columns A of the matrix,
            `A.sum(1) + bias_column(x) == f(x)` is an identity that a wrong
            linearization breaks. KnowledgeRowComputer uses it for its
            `independent_bias` option.

            A network split over devices (`NN.shard`) is followed along its
            device map, as in `forward`.

            Args:
                x (torch.Tensor): The input to the NN.
            Returns:
                torch.Tensor: The bias column, of shape (output_size,), on
                    the computer's device. The network output of the same
                    pass is in `current_output`.
        """
        with torch.no_grad():
            model = self.model
            model.save = True
            if model.layer_devices is None:
                model.to(self.device)
            try:
                self.current_output = model.forward(x)
            finally:
                model.save = False
            # NN.forward recorded the shape of the input after the start layers
            a = self._bias_pass((1, *model.input_shape), self.current_output.dtype)
            return a.reshape(-1).to(self.device)

    def _check_cuts(self, start_cut: int, end_cut: int) -> None:
        """Raise unless (start_cut, end_cut) is a valid pair of segment cuts."""
        model = self.model
        n = model.get_num_layers()
        if not (model._get_start_layer() <= start_cut < end_cut <= n):
            raise ValueError(f"invalid cuts ({start_cut}, {end_cut})")
        if model.concat_skips or model.branch_inputs:
            raise NotImplementedError(
                "segment() does not support networks with concatenation skips "
                "or branch inputs."
            )
        for end, (start, mid) in model.parallel_blocks.items():
            for c in (start_cut, end_cut):
                if start < c < end:
                    raise ValueError(f"cut {c} falls inside parallel block ({start},{end})")
        for end, pairs in model.residuals.items():
            for start, _ in pairs:
                for c in (start_cut, end_cut):
                    if start < c < end:
                        raise ValueError(f"cut {c} falls inside the residual block ({start},{end})")
        # Without chained_residuals, a block starting at a merge index skips
        # from the tensor BEFORE that merge, which lies upstream of the cut.
        if (not model.chained_residuals and start_cut in model.residuals
                and start_cut in model.residuals_starts):
            raise ValueError(
                f"cut {start_cut} is both the end and the start of a residual "
                f"block of a network without chained_residuals"
            )

    def segment(self, x: torch.Tensor, start_cut: int, end_cut: int,
                final_position_only: bool = False) -> torch.Tensor:
        """KM of layers [start_cut, end_cut) at input x, under the cut-point
        convention: the residual application AT start_cut belongs upstream (skipped);
        the residual application AT end_cut belongs to this segment (trailing step).
        Cut values are residual-stream tensors saved by the reference pass
        (`model.stream`, which this method records regardless of
        `model.save_stream`). The linearization follows mixer_mode /
        attention_mode, exactly as in forward()."""
        n = self.model.get_num_layers()
        self._check_cuts(start_cut, end_cut)
        if final_position_only and end_cut != n:
            raise ValueError("final_position_only requires end_cut == num_layers")

        with torch.no_grad():
            # Reference pass: activations, patterns, LN stats, stream values
            save_stream = self.model.save_stream
            if self.model.layer_devices is None:
                self.model.to(self.device)
            # (a network split over devices is never moved, see forward)
            self.model.save = True
            self.model.save_stream = True
            try:
                self.current_output = self.model.forward(x)
            finally:
                self.model.save = False
                self.model.save_stream = save_stream

            # The cut value lives on the device of layer start_cut, where
            # the probe batch of the segment starts.
            in_device = self._layer_device(start_cut)
            x0 = self.model.stream[start_cut].to(in_device)        # (1, C, T, D)
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
                branch_snapshots = [None] * n
                pstate = _ParallelState()
                step = self._affine_step if affine else self._linear_step
                for i, layer in enumerate(self.layers[start_cut:end_cut], start=start_cut):
                    P = self._wire(P, i, inputs_residuals, branch_snapshots, pstate,
                                   affine=affine, seg_start=start_cut)
                    P = step(P, i, layer)
                # Trailing step: the merges AT end_cut close this segment
                # (on the device of layer end_cut, where NN.forward does them).
                if end_cut < n:
                    P = self.model.to_layer_device(P, end_cut)
                if end_cut < n and end_cut in self.model.parallel_blocks:
                    P = pstate.saved.pop(end_cut) + pstate.branch1.pop(end_cut) + P
                if end_cut < n and end_cut in self.model.residuals:
                    P = self.model.apply_residual(P, inputs_residuals, layer=end_cut, affine=affine)
                return P

            for batch in range(num_batches):
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, total_positions)
                cbs = end - start
                indices = torch.arange(start, end, device=in_device)
                c = indices // (H * W)
                remaining = indices % (H * W)
                h = remaining // W
                w = remaining % W
                B = torch.zeros((cbs, C, H, W), device=in_device, dtype=dtype)
                B[torch.arange(cbs, device=in_device), c, h, w] = x0.flatten()[start:end]
                B = run_probe(B, affine=False)
                A[:, start:end] = B.reshape(cbs, -1)[:, row_lo:row_hi].T.to(self.device)

            a = torch.zeros(x0.shape, device=in_device, dtype=dtype)
            a = run_probe(a, affine=True)
            a = a.reshape(1, -1)[:, row_lo:row_hi]
            return torch.cat((A, a.T.to(self.device)), dim=-1)
