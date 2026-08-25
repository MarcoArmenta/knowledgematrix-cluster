from __future__ import annotations

import copy
import operator
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F
import torch.fx as fx
import math
from typing import Union, Dict, Tuple, List, Optional


class NN(nn.Module):
    """
        A class to build a neural network for which the knowledge matrix can be computed.

        Args:
            input_shape (Tuple[int]): The shape of the input to the network.
            save (bool): Whether to save the activations and preactivations of the network.
            device (str): The device to run the network on.
    """

    def __init__(
            self, 
            input_shape: Tuple[int],
            save: bool=False, 
            device: str="cpu"
        ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.save = save
        self.device = device
        self.layers = nn.ModuleList()
        self.residuals: Dict[int, Tuple[int, list[nn.Module]]] = {}
        self.residuals_starts: set[int] = set()
        self.residual_modules = nn.ModuleList()
        # Concatenation skip connections (used by DenseNet, U-Net):
        #   concat_skips[end] -> ordered list of source layer indices whose
        #   captured tensors are concatenated (along channel dim) BEFORE x at
        #   layer `end`. concat_skips_starts holds every source index so
        #   forward() snapshots x at the right moment.
        self.concat_skips: Dict[int, list[int]] = {}
        self.concat_skips_starts: set[int] = set()
        # Branch-input wiring (used by Inception): at layer `end`, REPLACE x
        # with the snapshot captured at layer `start` (i.e. discard whatever
        # value flowed through the previous branch). This linearizes the
        # parallel branches of an Inception module: branch 1 runs naturally,
        # branch_input restores x to the module's fork-point at the start of
        # branch 2, 3, ..., and the eventual concat at the merge layer is
        # handled by concat_skip.
        self.branch_inputs: Dict[int, int] = {}
        self.branch_inputs_starts: set[int] = set()


    ### Linear Layers ###

    def linear(
            self, 
            in_features: int, 
            out_features: int, 
            bias: bool=True
        ) -> None:
        self.layers.append(nn.Linear(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias
        ))
    
    def conv(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int],
            stride: Tuple[int]=(1,1),
            padding: Tuple[int]=(0,0),
            dilation: Tuple[int]=(1,1),
            groups: int=1,
            bias: bool=True
        ) -> None:
        self.layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        ))
    
    def conv1d(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int=1,
            padding: int=0,
            dilation: int=1,
            groups: int=1,
            bias: bool=True
        ) -> None:
        self.layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,1),
            stride=(stride,1),
            padding=(padding,0),
            dilation=(dilation,1),
            groups=groups,
            bias=bias
        ))

    def conv_transpose(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int],
            stride: Tuple[int]=(1,1),
            padding: Tuple[int]=(0,0),
            output_padding: Tuple[int]=(0,0),
            dilation: Tuple[int]=(1,1),
            groups: int=1,
            bias: bool=True
        ) -> None:
        self.layers.append(nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        ))

    def conv_transpose1d(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int=1,
            padding: int=0,
            output_padding: int=0,
            dilation: int=1,
            groups: int=1,
            bias: bool=True
        ) -> None:
        self.layers.append(nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,1),
            stride=(stride,1),
            padding=(padding,0),
            output_padding=(output_padding,0),
            dilation=(dilation,1),
            groups=groups,
            bias=bias
        ))
    
    def flatten(
            self,
            start_dim: int=1,
            end_dim: int=-1) -> None:
        self.layers.append(nn.Flatten(
            start_dim=start_dim,
            end_dim=end_dim
        ))
    
    def embedding(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Union[int,None]=None,
            max_norm: Union[float,None]=None,
            norm_type: float=2.0,
            scale_grad_by_freq: bool=False,
            sparse: bool=False
    ) -> None:
        self.layers.append(nn.Embedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse
        ))

    
    ### Normalization Layers ###

    def batchnorm(
            self,
            num_features: int,
            eps: float=0.00001,
            momentum: float=0.1
    ) -> None:
        self.layers.append(nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum
        ))
    
    def batchnorm1d(
            self,
            num_features: int,
            eps: float=0.00001,
            momentum: float=0.1
    ) -> None:
        self.batchnorm(num_features, eps, momentum)
    
    def layernorm(
            self,
            normalized_shape: Union[int,Tuple[int],torch.Size],
            eps: float=1e-5,
            elementwise_affine: bool=True,
            bias: bool=True
    ) -> None:
        self.layers.append(nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias
        ))

    def rmsnorm(self, normalized_shape: int, eps: float = 1e-6) -> None:
        self.layers.append(RMSNorm(normalized_shape=normalized_shape, eps=eps))

    def groupnorm(
            self,
            num_groups: int,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True
    ) -> None:
        self.layers.append(nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        ))

    def instancenorm(
            self,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True
    ) -> None:
        self.layers.append(nn.GroupNorm(
            num_groups=num_channels,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        ))


    ### Pooling Layers ###

    def avgpool(
            self,
            kernel_size: Tuple[int],
            stride: Union[Tuple[int],None]=None,
            padding: Tuple[int]=(0,0)
    ) -> None:
        if stride is None:
            stride = kernel_size
        self.layers.append(nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ))
    
    def avgpool1d(
            self,
            kernel_size: int,
            stride: Union[int,None]=None,
            padding: int=0
    ) -> None:
        if stride is None:
            stride = kernel_size
        self.layers.append(nn.AvgPool2d(
            kernel_size=(kernel_size,1),
            stride=(stride,1),
            padding=(padding,0)
        ))
    
    def adaptiveavgpool(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveAvgPool2d(output_size=output_size))
    
    def adaptiveavgpool1d(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveAvgPool2d(output_size=(output_size,1)))
    
    def maxpool(
            self, 
            kernel_size: Tuple[int],
            stride: Union[Tuple[int],None]=None,
            padding: Tuple[int]=(0,0)
        ) -> None:
        if stride is None:
            stride = kernel_size
        self.layers.append(nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True
        ))
    
    def maxpool1d(
            self, 
            kernel_size: int,
            stride: Union[int,None]=None,
            padding: int=0
        ) -> None:
        if stride is None:
            stride = kernel_size
        self.layers.append(nn.MaxPool2d(
            kernel_size=(kernel_size,1),
            stride=(stride,1),
            padding=(padding,0),
            return_indices=True
        ))
    
    def adaptivemaxpool(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveMaxPool2d(output_size=output_size, return_indices=True))

    def adaptivemaxpool1d(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveMaxPool2d(output_size=(output_size,1), return_indices=True))


    ### Upsampling Layers ###

    def upsample(
            self,
            scale_factor: Union[int, float, Tuple[int]],
            mode: str = 'nearest'
    ) -> None:
        kwargs = {'scale_factor': scale_factor, 'mode': mode}
        if mode in ('bilinear', 'bicubic', 'trilinear'):
            kwargs['align_corners'] = False
        self.layers.append(nn.Upsample(**kwargs))

    def pixel_shuffle(self, upscale_factor: int) -> None:
        self.layers.append(nn.PixelShuffle(upscale_factor=upscale_factor))


    ### Dropout ###

    def dropout(self, p: float=0.5) -> None:
        self.layers.append(nn.Dropout(p=p))

    def identity(self) -> None:
        """
            A no-op layer. Useful as a boundary for residual wiring: a layer
            index that is simultaneously the end of one residual and the
            start of the next captures its snapshot BEFORE the addition is
            applied (starts are snapshotted before ends are applied). In
            pre-norm architectures (x = x + f(norm(x)) chained), insert an
            identity after each addition point so the next residual's start
            index differs from the previous residual's end index.
        """
        self.layers.append(nn.Identity())


    ### Activation Functions ###

    def elu(self, alpha: float=1) -> None:
        self.layers.append(nn.ELU(alpha=alpha))

    def gelu(self, approximate: str="none") -> None:
        self.layers.append(nn.GELU(approximate=approximate))

    def leakyrelu(self, negative_slope: float=0.01) -> None:
        self.layers.append(nn.LeakyReLU(negative_slope=negative_slope))

    def relu(self) -> None:
        self.layers.append(nn.ReLU())
    
    def sigmoid(self) -> None:
        self.layers.append(nn.Sigmoid())

    def silu(self) -> None:
        self.layers.append(nn.SiLU())

    def mish(self) -> None:
        self.layers.append(nn.Mish())

    def softmax(self, dim: Union[int,None]=None) -> None:
        self.layers.append(nn.Softmax(dim=dim))

    def tanh(self) -> None:
        self.layers.append(nn.Tanh())

    def celu(self, alpha: float = 1.0) -> None:
        self.layers.append(nn.CELU(alpha=alpha))

    def hardsigmoid(self) -> None:
        self.layers.append(nn.Hardsigmoid())

    def hardswish(self) -> None:
        self.layers.append(nn.Hardswish())

    def prelu(self, num_parameters: int = 1, init: float = 0.25) -> None:
        self.layers.append(nn.PReLU(num_parameters=num_parameters, init=init))

    def relu6(self) -> None:
        self.layers.append(nn.ReLU6())

    def softplus(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        self.layers.append(nn.Softplus(beta=beta, threshold=threshold))

    def jumprelu(self, threshold: torch.Tensor) -> None:
        self.layers.append(JumpReLU(threshold=threshold))

    def topk_activation(self, k: int) -> None:
        self.layers.append(TopKActivation(k=k))

    def multiheadattention(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: Union[int,None]=None,
            mask: Union[torch.Tensor,None]=None
        ) -> None:
        self.layers.append(
            MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mask=mask
            )
        )


    def gatedattention(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: Union[int, None]=None,
            head_dim: Union[int, None]=None,
            rope_theta: float=10000.0,
            partial_rotary_factor: float=0.25,
            rms_norm_eps: float=1e-6,
            bias: bool=False,
            causal: bool=True
        ) -> None:
        self.layers.append(
            GatedAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rope_theta=rope_theta,
                partial_rotary_factor=partial_rotary_factor,
                rms_norm_eps=rms_norm_eps,
                bias=bias,
                causal=causal
            )
        )

    def gateddeltanet(
            self,
            d_model: int,
            num_v_heads: int,
            num_k_heads: int,
            head_k_dim: int=128,
            head_v_dim: int=128,
            conv_kernel_size: int=4,
            rms_norm_eps: float=1e-6
        ) -> None:
        self.layers.append(
            GatedDeltaNet(
                d_model=d_model,
                num_v_heads=num_v_heads,
                num_k_heads=num_k_heads,
                head_k_dim=head_k_dim,
                head_v_dim=head_v_dim,
                conv_kernel_size=conv_kernel_size,
                rms_norm_eps=rms_norm_eps
            )
        )

    def swiglu(self, d_model: int, d_ff: int, bias: bool=False) -> None:
        self.layers.append(SwiGLU(d_model=d_model, d_ff=d_ff, bias=bias))


    ### Positional Encoding ###

    def positionalencoding(self, d_model: int, max_len: int=5000) -> None:
        self.layers.append(PositionalEncoding(d_model, max_len))


    ### Residual Connections ###

    def residual(self, start: int, end: int) -> None:
        shape_start = self.shape_at_layer(start)
        shape_end = self.shape_at_layer(end)
        if (start < end):
            if shape_start == shape_end:
                projection = [nn.Identity()]
            else:
                if len(shape_start) == len(shape_end):
                    if len(shape_start) >= 4 and len(shape_end) >= 4:  # Conv
                        projection = [
                            nn.Conv2d(
                                shape_start[1], 
                                shape_end[1], 
                                kernel_size=1,
                                stride = (
                                    round(shape_start[2] / shape_end[2]),
                                    round(shape_start[3] / shape_end[3]),
                                ),
                                bias=False
                            ),
                            nn.BatchNorm2d(shape_end[1])
                        ]
                    elif len(shape_start) <= 3 and len(shape_end) <= 3:  # FC
                        projection = [
                            nn.Linear(
                                shape_start[-1],
                                shape_end[-1],
                                bias=True
                            )
                        ]
                else:
                    raise ValueError(f"The lenghts of shape at layer {start} and {end} need to be equal to have a residual connection. Got {shape_start} and {shape_end}.")
        else:
            raise ValueError(f"To have a residual connection from layer {start} to {end}, one needs {start} < {end}.")
        for module in projection:
            if not isinstance(module, nn.Identity):
                self.residual_modules.append(module)
        self.residuals_starts.add(start)
        if end in self.residuals:
                self.residuals[end].append((start, projection))
        else:
            self.residuals[end] = [(start, projection)]

    def concat_skip(self, start: int, end: int) -> None:
        """
            Register a channel-wise concatenation skip: the tensor captured
            at `start` is concatenated with the tensor arriving at `end`
            along dim=1 (channels), in call order with the current x last.
            Sources must share spatial dims with x at `end`. No projection
            is applied -- the user is responsible for sizing downstream
            layers to receive the grown channel dim.
        """
        if start >= end:
            raise ValueError(f"concat_skip requires start < end, got start={start}, end={end}.")
        self.concat_skips_starts.add(start)
        if end in self.concat_skips:
            self.concat_skips[end].append(start)
        else:
            self.concat_skips[end] = [start]

    def apply_concat(self, x: torch.Tensor, outputs: list[torch.Tensor], layer: int) -> torch.Tensor:
        sources = [outputs[s] for s in self.concat_skips[layer]]
        return torch.cat(sources + [x], dim=1)

    def branch_input(self, start: int, end: int) -> None:
        """
            Register a branch-input wiring: at layer `end`, REPLACE x with
            the tensor captured at layer `start`. Used to linearize the
            parallel branches of an Inception module -- after branch k
            finishes, branch k+1 starts from the fork-point snapshot rather
            than from branch k's output. The current x at `end` is dropped
            (its value is typically captured separately as branch k's output
            via concat_skip's start-snapshot mechanism, since `end` will
            usually be in concat_skips_starts too).
        """
        if start >= end:
            raise ValueError(f"branch_input requires start < end, got start={start}, end={end}.")
        if end in self.branch_inputs:
            raise ValueError(f"branch_input destination {end} already set to {self.branch_inputs[end]}.")
        self.branch_inputs_starts.add(start)
        self.branch_inputs[end] = start


    ### Conversion from an arbitrary torch Module (MVP) ###

    @classmethod
    def from_torch(
            cls,
            module: nn.Module,
            input_shape: Tuple[int, ...],
            num_classes: Union[int, None] = None,
            device: str = "cpu",
        ) -> "NN":
        """
            Convert an in-scope pretrained torchvision ``nn.Module`` into an
            ``NN`` whose ``forward`` exactly reproduces the source (so its
            knowledge matrix can be computed).

            MVP scope: pure-sequential nets (VGG, AlexNet) and post-activation
            residual nets that merge via ``operator.add`` with a post-add
            activation (ResNet-18/34/50/101/152). Anything else -- concat
            (DenseNet), branch (Inception), SE/``mul`` gating, attention,
            reshape/permute, linear-bottleneck/pre-activation residuals -- is
            out of scope and raises ``NotImplementedError`` rather than
            silently mis-wiring.

            Pipeline: ``torch.fx.symbolic_trace`` -> classify each node
            (LAYER / WIRING / DROP / UNMAPPABLE) -> emit into a fresh ``NN``,
            resolving residual skip endpoints to integer layer indices via a
            ``live_index`` map (first emitted-layer consumer of a tensor).

            Args:
                module: the source ``nn.Module`` (left untouched -- all
                    parametric submodules are deep-copied).
                input_shape: ``(C, H, W)`` shape of a single input sample.
                num_classes: if given and the final ``Linear`` has a different
                    ``out_features``, the head is replaced with a fresh
                    ``Linear`` of this width (no pretrained weights).
                device: device for the resulting ``NN``.
        """
        gm = fx.symbolic_trace(module)
        nodes = list(gm.graph.nodes)

        # ---- Pass 0: find residual-merge nodes and their skip/downsample paths.
        merge_nodes: List[fx.Node] = [
            n for n in nodes
            if _ft_is_add(n) and len(_ft_tensor_inputs(n)) == 2
        ]
        merge_set = set(merge_nodes)
        merge_info: Dict[fx.Node, Tuple[fx.Node, List[nn.Module]]] = {}
        skip_node_set: set = set()
        for m in merge_nodes:
            a, b = _ft_tensor_inputs(m)
            fork = _ft_fork(a, b)
            if fork is None:
                raise NotImplementedError(
                    f"from_torch: cannot resolve a common fork for residual add "
                    f"at node {m.name}."
                )
            da, db = _ft_dist(a, fork), _ft_dist(b, fork)
            # main = the deeper branch; skip = the shallower (identity/downsample).
            skip_arg = b if da >= db else a
            skip_path = _ft_linear_path(skip_arg, fork, m)
            skip_src: List[nn.Module] = []
            for sn in skip_path:
                if sn.op != "call_module":
                    raise NotImplementedError(
                        f"from_torch: residual skip path at node {m.name} contains a "
                        f"non-module op {sn.op} ({sn.target}); out of MVP scope."
                    )
                sub = gm.get_submodule(sn.target)
                if not isinstance(sub, (nn.Conv2d, nn.BatchNorm2d)):
                    raise NotImplementedError(
                        f"from_torch: unsupported downsample/projection module "
                        f"{type(sub).__name__} at node {sn.name}; out of MVP scope."
                    )
                skip_src.append(sub)
            merge_info[m] = (fork, skip_src)
            skip_node_set.update(skip_path)

        # ---- Pass 1: emit layers; build emit_index / live_index / passthrough.
        self = cls(input_shape, save=False, device=device)
        live_index: Dict[fx.Node, int] = {}
        passthrough: Dict[fx.Node, fx.Node] = {}

        def resolve(n: fx.Node) -> fx.Node:
            seen = set()
            while n in passthrough and n not in seen:
                seen.add(n)
                n = passthrough[n]
            return n

        for node in nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node in skip_node_set or node in merge_set:
                # Skip/downsample modules and residual adds are handled as
                # wiring in pass 2, not emitted on the main line.
                continue
            status = self._ft_emit(node, gm)
            if status == "drop":
                ins = _ft_tensor_inputs(node)
                passthrough[node] = resolve(ins[0]) if ins else node
                continue
            # A layer was appended; record its index and its consumption of
            # upstream tensors (first emitted consumer wins -> live_index).
            idx = self.get_num_layers() - 1
            for inp in _ft_tensor_inputs(node):
                src = resolve(inp)
                if src not in live_index:
                    live_index[src] = idx

        # ---- Pass 2: register residual wiring; guard the two hazards.
        for m in merge_nodes:
            fork, skip_src = merge_info[m]
            fork_r = resolve(fork)
            # Hazard 1: terminal add (no post-merge layer consumes it).
            if m not in live_index:
                raise NotImplementedError(
                    f"from_torch: terminal residual add at node {m.name} (no "
                    f"following layer); NN.forward would silently drop it."
                )
            end = live_index[m]
            # Hazard 2: no-gap adjacent merge. The fork must resolve to an
            # emitted-layer output (or the network input), never a raw merge
            # output -- otherwise the fork snapshot captures the pre-add tensor.
            if fork_r in merge_set:
                raise NotImplementedError(
                    f"from_torch: residual at node {m.name} forks from an "
                    f"un-activated merge output (no post-merge activation between "
                    f"adjacent residual blocks). Architectures such as "
                    f"MobileNetV2/EfficientNet linear bottlenecks and "
                    f"pre-activation ResNets are out of MVP scope."
                )
            if fork_r not in live_index:
                raise NotImplementedError(
                    f"from_torch: cannot resolve residual fork index for node "
                    f"{m.name}."
                )
            start = live_index[fork_r]
            if not (start < end) or end >= self.get_num_layers():
                raise NotImplementedError(
                    f"from_torch: invalid residual endpoints (start={start}, "
                    f"end={end}) at node {m.name}."
                )
            self.residual(start, end)
            # Keep auto-projection BNs in eval so later shape_at_layer calls
            # don't corrupt their running stats.
            for _, proj in self.residuals[end]:
                for sub in proj:
                    if isinstance(sub, nn.BatchNorm2d):
                        sub.eval()
            # Override the auto projection with the real downsample modules.
            if skip_src:
                new_proj = [copy.deepcopy(s) for s in skip_src]
                for mod in new_proj:
                    if isinstance(mod, nn.BatchNorm2d):
                        mod.eval()
                    self.residual_modules.append(mod)
                overridden = False
                lst = self.residuals[end]
                for k, (s, _proj) in enumerate(lst):
                    if s == start:
                        lst[k] = (start, new_proj)
                        overridden = True
                        break
                assert overridden, (
                    f"from_torch: downsample projection override failed for node "
                    f"{m.name} (start={start}, end={end})."
                )

        # ---- Optional classifier-head swap.
        if num_classes is not None:
            last = self.layers[-1]
            if isinstance(last, nn.Linear) and last.out_features != num_classes:
                self.layers[-1] = nn.Linear(last.in_features, num_classes)

        self.to(device)
        return self

    def _ft_emit(self, node: fx.Node, gm: fx.GraphModule) -> str:
        """
            Emit one fx node into ``self.layers``. Returns "layer" if a layer
            was appended, "drop" if the node is a no-op at eval (Dropout /
            Identity). Raises ``NotImplementedError`` for anything unmappable.
        """
        if node.op == "call_module":
            return self._ft_emit_module(gm.get_submodule(node.target), node)
        if node.op in ("call_function", "call_method"):
            return self._ft_emit_func(node)
        raise NotImplementedError(
            f"from_torch: unsupported node op {node.op} at node {node.name}."
        )

    def _ft_emit_module(self, sub: nn.Module, node: fx.Node) -> str:
        # Drop no-ops (identity at eval).
        if isinstance(sub, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                            nn.Dropout3d, nn.AlphaDropout, nn.Identity)):
            return "drop"
        # Parametric: deep-copy so weights/bias/BN stats transfer exactly and
        # the caller's module is never mutated.
        if isinstance(sub, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                            nn.LayerNorm, nn.GroupNorm)):
            self.layers.append(copy.deepcopy(sub))
            return "layer"
        if isinstance(sub, nn.BatchNorm2d):
            bn = copy.deepcopy(sub)
            bn.eval()  # protect running stats from shape_at_layer's random forwards
            self.layers.append(bn)
            return "layer"
        if isinstance(sub, nn.PReLU):  # parametric activation
            self.layers.append(copy.deepcopy(sub))
            return "layer"
        # Stateless ops rebuilt via the NN builders.
        if isinstance(sub, nn.ReLU):
            self.relu(); return "layer"
        if isinstance(sub, nn.ReLU6):
            self.relu6(); return "layer"
        if isinstance(sub, nn.LeakyReLU):
            self.leakyrelu(sub.negative_slope); return "layer"
        if isinstance(sub, nn.ELU):
            self.elu(sub.alpha); return "layer"
        if isinstance(sub, nn.CELU):
            self.celu(sub.alpha); return "layer"
        if isinstance(sub, nn.SiLU):
            self.silu(); return "layer"
        if isinstance(sub, nn.GELU):
            self.gelu(sub.approximate); return "layer"
        if isinstance(sub, nn.Mish):
            self.mish(); return "layer"
        if isinstance(sub, nn.Sigmoid):
            self.sigmoid(); return "layer"
        if isinstance(sub, nn.Hardsigmoid):
            self.hardsigmoid(); return "layer"
        if isinstance(sub, nn.Hardswish):
            self.hardswish(); return "layer"
        if isinstance(sub, nn.Tanh):
            self.tanh(); return "layer"
        if isinstance(sub, nn.Softplus):
            self.softplus(sub.beta, sub.threshold); return "layer"
        if isinstance(sub, nn.Softmax):
            self.softmax(sub.dim); return "layer"
        if isinstance(sub, nn.MaxPool2d):
            self.maxpool(sub.kernel_size, sub.stride, sub.padding); return "layer"
        if isinstance(sub, nn.AdaptiveMaxPool2d):
            self.adaptivemaxpool(sub.output_size); return "layer"
        if isinstance(sub, nn.AvgPool2d):
            self.avgpool(sub.kernel_size, sub.stride, sub.padding); return "layer"
        if isinstance(sub, nn.AdaptiveAvgPool2d):
            self.adaptiveavgpool(sub.output_size); return "layer"
        if isinstance(sub, nn.Flatten):
            self.flatten(sub.start_dim, sub.end_dim); return "layer"
        raise NotImplementedError(
            f"from_torch: unsupported module {type(sub).__name__} at node "
            f"{node.name}."
        )

    def _ft_emit_func(self, node: fx.Node) -> str:
        tgt = node.target
        is_method = node.op == "call_method"
        # Flatten.
        if tgt in (torch.flatten,) or (is_method and tgt == "flatten"):
            start_dim = _ft_arg(node, 1, "start_dim", 1)
            end_dim = _ft_arg(node, 2, "end_dim", -1)
            self.flatten(start_dim, end_dim); return "layer"
        # Functional activations.
        if tgt in (F.relu, torch.relu) or (is_method and tgt in ("relu", "relu_")):
            self.relu(); return "layer"
        if tgt is F.relu6:
            self.relu6(); return "layer"
        if tgt is F.leaky_relu:
            self.leakyrelu(_ft_arg(node, 1, "negative_slope", 0.01)); return "layer"
        if tgt is F.elu:
            self.elu(_ft_arg(node, 1, "alpha", 1.0)); return "layer"
        if tgt is F.silu:
            self.silu(); return "layer"
        if tgt is F.gelu:
            self.gelu(_ft_arg(node, 1, "approximate", "none")); return "layer"
        if tgt is F.mish:
            self.mish(); return "layer"
        if tgt in (F.sigmoid, torch.sigmoid) or (is_method and tgt == "sigmoid"):
            self.sigmoid(); return "layer"
        if tgt is F.hardsigmoid:
            self.hardsigmoid(); return "layer"
        if tgt is F.hardswish:
            self.hardswish(); return "layer"
        if tgt in (F.tanh, torch.tanh) or (is_method and tgt == "tanh"):
            self.tanh(); return "layer"
        if tgt is F.softmax or (is_method and tgt == "softmax"):
            self.softmax(_ft_arg(node, 1, "dim", None)); return "layer"
        # Functional pooling.
        if tgt is F.max_pool2d:
            self.maxpool(_ft_arg(node, 1, "kernel_size"),
                         _ft_arg(node, 2, "stride", None),
                         _ft_arg(node, 3, "padding", 0)); return "layer"
        if tgt is F.avg_pool2d:
            self.avgpool(_ft_arg(node, 1, "kernel_size"),
                         _ft_arg(node, 2, "stride", None),
                         _ft_arg(node, 3, "padding", 0)); return "layer"
        if tgt is F.adaptive_avg_pool2d:
            self.adaptiveavgpool(_ft_arg(node, 1, "output_size")); return "layer"
        if tgt is F.adaptive_max_pool2d:
            self.adaptivemaxpool(_ft_arg(node, 1, "output_size")); return "layer"
        raise NotImplementedError(
            f"from_torch: unsupported op {tgt} at node {node.name}."
        )


    ### Forward Method ###

    def forward(self, x: torch.Tensor, return_penultimate:bool=False) -> torch.Tensor:
        start_layer = self._get_start_layer()  # Start layer is the one after the embedding and positional encoding
        for layer in self.layers[:start_layer]:
            x = layer(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # Update the input shape, useful when the input shape is not known beforehand (e.g. for transformers)
        self.input_shape = (x.shape[1], x.shape[2], x.shape[3])
        inputs_residuals: list[torch.Tensor] = [None] * self.get_num_layers()
        # branch_snapshots holds POST-concat snapshots for branch_input sources;
        # inputs_residuals holds PRE-concat snapshots for residual / concat_skip sources.
        branch_snapshots: list[torch.Tensor] = [None] * self.get_num_layers()
        if not self.save:  # Regular forward pass
            layers = self.layers[:-1] if return_penultimate else self.layers
            for i, layer in enumerate(layers[start_layer:], start=start_layer):
                if i in self.residuals_starts or i in self.concat_skips_starts:
                    inputs_residuals[i] = x
                if i in self.branch_inputs:
                    x = branch_snapshots[self.branch_inputs[i]]
                if i in self.residuals:
                    x = self.apply_residual(x, inputs_residuals, layer=i)
                if i in self.concat_skips:
                    x = self.apply_concat(x, inputs_residuals, layer=i)
                if i in self.branch_inputs_starts:
                    branch_snapshots[i] = x
                if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                    x, _ = layer(x)
                else:
                    x = layer(x)
        else:  # Forward pass for matrix computation
               # Save activations and preactivations
            if return_penultimate:
                raise ValueError("return_penultimate is not supported for matrix computation.")
            self.pre_acts: list[torch.Tensor] = [None] * self.get_num_layers()
            self.acts: list[torch.Tensor] = [None] * self.get_num_layers()
            self.maxpool_indices: list[torch.Tensor] = [None] * self.get_num_layers()
            self.layernorms: list[torch.Tensor] = [None] * self.get_num_layers()

            for i, layer in enumerate(self.layers[start_layer:], start=start_layer):
                if i in self.residuals_starts or i in self.concat_skips_starts:
                    inputs_residuals[i] = x
                if i in self.branch_inputs:
                    x = branch_snapshots[self.branch_inputs[i]]
                if i in self.residuals:
                    x = self.apply_residual(x, inputs_residuals, layer=i)
                if i in self.concat_skips:
                    x = self.apply_concat(x, inputs_residuals, layer=i)
                if i in self.branch_inputs_starts:
                    branch_snapshots[i] = x
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Linear, nn.Flatten, nn.Upsample, nn.PixelShuffle)):
                    x = layer(x)
                elif isinstance(layer, nn.LayerNorm):
                    dims = tuple(range(-len(layer.normalized_shape), 0))
                    self.layernorms[i] = (torch.mean(x, dim=dims, keepdim=True), torch.var(x, dim=dims, unbiased=False, keepdim=True))
                    x = layer(x)
                elif isinstance(layer, RMSNorm):
                    self.layernorms[i] = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + layer.eps)
                    x = layer(x)
                elif isinstance(layer, nn.GroupNorm):
                    G = layer.num_groups
                    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
                    cpg = C // G
                    x_grouped = x.reshape(N, G, cpg, H, W)
                    mean = x_grouped.mean(dim=[2, 3, 4], keepdim=True)
                    var = x_grouped.var(dim=[2, 3, 4], unbiased=False, keepdim=True)
                    mean_expanded = mean.expand(-1, -1, cpg, -1, -1).reshape(N, C, 1, 1)
                    var_expanded = var.expand(-1, -1, cpg, -1, -1).reshape(N, C, 1, 1)
                    self.layernorms[i] = (mean_expanded, var_expanded)
                    x = layer(x)
                elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                    x, indices = layer(x)
                    self.maxpool_indices[i] = indices
                elif isinstance(layer, ACTIVATION_LAYERS):
                    self.pre_acts[i] = x.detach().clone()
                    x = layer(x)
                    self.acts[i] = x.detach().clone()
        return x

    def apply_residual(self, x: torch.Tensor, outputs: list[torch.Tensor], layer: int, affine: bool=True) -> torch.Tensor:
        if affine:
            for start_idx, proj in self.residuals[layer]:
                output = outputs[start_idx]
                for layer in proj:
                    output = layer(output)
                x = x + output
        else:
            for start_idx, proj in self.residuals[layer]:
                output = outputs[start_idx]
                for layer in proj:
                    if isinstance(layer, nn.BatchNorm2d):
                        output = output * (layer.weight/torch.sqrt(layer.running_var+layer.eps)).view(1,-1,1,1)
                    elif isinstance(layer, nn.Linear):
                        output = torch.matmul(layer.weight, output.T).T
                    elif isinstance(layer, nn.Conv2d):
                        output = F.conv2d(output, layer.weight, None, stride=layer.stride, padding=layer.padding)
                    else:
                        output = layer(output)
                x = x + output
        return x


    ### Useful Functions ###

    def shape_at_layer(self, i: int) -> torch.Size:
        x = torch.randn(self.input_shape).unsqueeze(0)
        start_layer = self._get_start_layer()
        inputs_residuals: list[torch.Tensor] = [None] * self.get_num_layers()
        branch_snapshots: list[torch.Tensor] = [None] * self.get_num_layers()
        for j, layer in enumerate(self.layers[start_layer:i], start=start_layer):
            if j in self.concat_skips_starts:
                inputs_residuals[j] = x
            if j in self.branch_inputs:
                x = branch_snapshots[self.branch_inputs[j]]
            if j in self.concat_skips:
                x = self.apply_concat(x, inputs_residuals, layer=j)
            if j in self.branch_inputs_starts:
                branch_snapshots[j] = x
            if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                x, _ = layer(x)
            else:
                x = layer(x)
        return x.shape

    def get_matrix_shape(self) -> Tuple[int]:
        # Returns the shape of the knowledge matrix in the format: (rows, columns).
        return (self.layers[-1].out_features, self.get_input_size() + int(self._has_bias() or self._has_batchnorm() or self._has_layernorm() or self._has_groupnorm()))
    
    def _has_bias(self) -> bool:
        for layer in self.layers:
            try: 
                _ = layer.bias.data
                return True
            except:
                continue
        return False

    def _has_batchnorm(self) -> bool:
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm2d):
                return True
        return False

    def _has_layernorm(self) -> bool:
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                return True
        return False

    def _has_groupnorm(self) -> bool:
        for layer in self.layers:
            if isinstance(layer, nn.GroupNorm):
                return True
        return False

    def get_input_size(self) -> int:
        input_size = 1
        for i in self.input_shape:
            input_size *= i
        return input_size
    
    def get_num_layers(self) -> int:
        return len(self.layers)

    def eval(self) -> None:
        for layer in self.layers:
            layer.eval()
        for end in self.residuals:
            for _, proj in self.residuals[end]:
                for layer in proj:
                    layer.eval()

    def train(self) -> None:
        for layer in self.layers:
            layer.train()
        for end in self.residuals:
            for _, proj in self.residuals[end]:
                for layer in proj:
                    layer.train()

    def freeze(self) -> None:
        # Puts requires_grad = False to all parameters of all layers
        self._freeze_or_unfreeze(freeze=True)
    
    def freeze_at_layer(self, layer: int) -> None:
        # Puts requires_grad = False to all parameters of the specified layer
        for param in self.layers[layer].parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        # Puts requires_grad = True to all parameters of all layers
        self._freeze_or_unfreeze(freeze=False)
    
    def unfreeze_at_layer(self, layer: int) -> None:
        # Puts requires_grad = True to all parameters of the specified layer
        for param in self.layers[layer].parameters():
            param.requires_grad = True

    def _freeze_or_unfreeze(self, freeze: bool=True) -> None:
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = not freeze
        for end in self.residuals:
            for _, proj in self.residuals[end]:
                for layer in proj:
                    for param in layer.parameters():
                        param.requires_grad = not freeze

    def _get_start_layer(self) -> int:
        start_layer = 0
        if isinstance(self.layers[0], nn.Embedding):
            start_layer = 1
            if isinstance(self.layers[1], PositionalEncoding):
                start_layer = 2
        return start_layer


### Helpers for NN.from_torch (torch.fx graph analysis) ###

_FT_ADD_FUNCS = {operator.add, operator.iadd, torch.add}


def _ft_is_add(node: fx.Node) -> bool:
    """True if ``node`` is a tensor-add (candidate residual merge)."""
    if node.op == "call_function" and node.target in _FT_ADD_FUNCS:
        return True
    if node.op == "call_method" and node.target == "add":
        return True
    return False


def _ft_tensor_inputs(node: fx.Node) -> List[fx.Node]:
    """The fx.Node tensor producers consumed by ``node`` (args then kwargs)."""
    out = [a for a in node.args if isinstance(a, fx.Node)]
    out += [v for v in node.kwargs.values() if isinstance(v, fx.Node)]
    return out


def _ft_ancestors(node: fx.Node) -> set:
    """All transitive tensor-producing ancestors of ``node`` (excluding it)."""
    seen: set = set()
    stack = [node]
    while stack:
        n = stack.pop()
        for inp in _ft_tensor_inputs(n):
            if inp not in seen:
                seen.add(inp)
                stack.append(inp)
    return seen


def _ft_fork(a: fx.Node, b: fx.Node) -> Optional[fx.Node]:
    """
        Lowest common ancestor of two add-inputs: the tensor where the main
        and skip branches diverge. Found by walking back from ``b`` (closest
        first) until reaching a node that is also an ancestor of ``a``.
    """
    anc_a = _ft_ancestors(a)
    anc_a.add(a)
    dq = deque([b])
    visited: set = set()
    while dq:
        n = dq.popleft()
        if n in anc_a:
            return n
        if n in visited:
            continue
        visited.add(n)
        for inp in _ft_tensor_inputs(n):
            dq.append(inp)
    return None


def _ft_dist(node: fx.Node, target: fx.Node) -> Optional[int]:
    """Shortest edge distance from ``node`` back to ``target`` (0 if equal)."""
    dq = deque([(node, 0)])
    visited: set = set()
    while dq:
        n, d = dq.popleft()
        if n is target:
            return d
        if n in visited:
            continue
        visited.add(n)
        for inp in _ft_tensor_inputs(n):
            dq.append((inp, d + 1))
    return None


def _ft_linear_path(skip_arg: fx.Node, fork: fx.Node, merge: fx.Node) -> List[fx.Node]:
    """
        Ordered list of nodes on the skip branch from just-after ``fork`` to
        ``skip_arg`` (inclusive), i.e. the downsample/projection chain. Empty
        for an identity skip (``skip_arg is fork``). Requires the chain to be
        linear (single tensor input per hop); otherwise raises.
    """
    path: List[fx.Node] = []
    cur = skip_arg
    while cur is not fork:
        path.append(cur)
        ins = _ft_tensor_inputs(cur)
        if len(ins) != 1:
            raise NotImplementedError(
                f"from_torch: non-linear residual skip branch feeding node "
                f"{merge.name}; out of MVP scope."
            )
        cur = ins[0]
    path.reverse()
    return path


def _ft_arg(node: fx.Node, idx: int, key: str, default=None):
    """Read a positional-or-keyword fx call argument with a fallback."""
    if key in node.kwargs:
        return node.kwargs[key]
    if idx < len(node.args):
        return node.args[idx]
    return default


class RMSNorm(nn.Module):
    """
        Root Mean Square Layer Normalization.
        Used by LLaMA, Mistral, Gemma, Qwen, and most post-2023 LLMs.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * self.weight / rms
   

class RMSNormGated(nn.Module):
    """
        RMS normalization followed by a SiLU gate, as used inside the
        GatedDeltaNet layers of Qwen3-Next / Qwen3.5:
        out = (x / rms(x)) * weight * silu(z).

        Args:
            hidden_size (int): The size of the normalized (last) dimension.
            eps (float): Numerical stability constant added to the mean square.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight * F.silu(gate)


class SwiGLU(nn.Module):
    """
        Gated feed-forward network (SwiGLU), the MLP block of LLaMA / Qwen
        and most post-2023 LLMs:
        out = down_proj( silu(gate_proj(x)) * up_proj(x) ).

        The whole block maps d_model -> d_model, so the knowledge matrix
        computation treats it like an activation (elementwise post/pre
        ratio), exactly as it does for MultiHeadAttention.

        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the hidden (intermediate) layer.
            bias (bool): Whether the three projections have a bias.
    """
    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def frozen_forward(self, x: torch.Tensor, x0: torch.Tensor, affine: bool = True) -> torch.Tensor:
        """
            Frozen-gate linearization of the block, evaluated at x: the
            silu(gate_proj) factor is computed from (and frozen at) the
            actual layer input x0, so the map x -> frozen_forward(x, x0)
            is LINEAR (through the up_proj path) and equals forward(x0) at
            x = x0 exactly. Used by KnowledgeMatrixComputer(mixer_mode=
            "frozen") and KnowledgeRowComputer. With affine=False the
            up/down projections are applied without their biases (the
            knowledge matrix column pass).
        """
        gate0 = F.silu(self.gate_proj(x0))
        up_bias = self.up_proj.bias if (affine and self.up_proj.bias is not None) else None
        down_bias = self.down_proj.bias if (affine and self.down_proj.bias is not None) else None
        return F.linear(gate0 * F.linear(x, self.up_proj.weight, up_bias), self.down_proj.weight, down_bias)


class GatedAttention(nn.Module):
    """
        Full attention block of Qwen3-Next / Qwen3.5 ("gated attention"):
        - fused query + output-gate projection (q_proj outputs
          num_heads * head_dim * 2, split per head into query and gate),
        - RMS normalization of queries and keys per head (QK-Norm),
        - partial rotary position embedding (RoPE applied to the first
          head_dim * partial_rotary_factor dimensions of each head),
        - grouped-query attention (num_kv_heads < num_heads),
        - causal masking,
        - sigmoid output gate before the output projection.

        Matches the reference implementation in Hugging Face transformers
        (models/qwen3_5, Apache-2.0). The zero-centered RMSNorm weights of
        the checkpoint ((1 + w) scaling) must be folded into plain weights
        (w' = 1 + w) when loading, which models/qwen3_5.py does.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of query heads.
            num_kv_heads (int): The number of key/value heads.
            head_dim (int): The dimension of each head (need not equal
                d_model // num_heads in Qwen3.5).
            rope_theta (float): The RoPE base frequency.
            partial_rotary_factor (float): The fraction of each head that
                is rotated by RoPE.
            rms_norm_eps (float): Epsilon of the q/k RMS normalization.
            bias (bool): Whether the q/k/v/o projections have a bias.
            causal (bool): Whether to apply a causal mask.
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: Union[int, None] = None,
            head_dim: Union[int, None] = None,
            rope_theta: float = 10000.0,
            partial_rotary_factor: float = 0.25,
            rms_norm_eps: float = 1e-6,
            bias: bool = False,
            causal: bool = True
        ) -> None:
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        if head_dim is None:
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads when head_dim is not given.")
            head_dim = d_model // num_heads

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = head_dim
        self.kv_repeat = num_heads // num_kv_heads
        self.rope_theta = rope_theta
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        if self.rotary_dim % 2 != 0:
            raise ValueError("head_dim * partial_rotary_factor must be even.")
        self.causal = causal

        self.q_proj = nn.Linear(d_model, num_heads * head_dim * 2, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=bias)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def _rope(self, T: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # cos/sin of shape (T, rotary_dim), computed in float32 like the
        # reference implementation (or float64 when running in float64).
        rdt = torch.float64 if dtype == torch.float64 else torch.float32
        half = self.rotary_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=rdt, device=device) / self.rotary_dim)
        )
        pos = torch.arange(T, dtype=rdt, device=device)
        freqs = torch.outer(pos, inv_freq)  # (T, rotary_dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, rotary_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (..., T, head_dim); rotate the first rotary_dim dims only.
        x_rot, x_pass = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        x_rot = (x_rot * cos) + (self._rotate_half(x_rot) * sin)
        return torch.cat((x_rot, x_pass), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, B, T, D = x.shape

        q = self.q_proj(x).view(batch, B, T, self.num_heads, self.d_head * 2)
        q, gate = torch.chunk(q, 2, dim=-1)  # per-head split, like the reference
        gate = gate.reshape(batch, B, T, self.num_heads * self.d_head)

        q = self.q_norm(q).transpose(2, 3)  # (batch, B, H, T, d_head)
        k = self.k_norm(self.k_proj(x).view(batch, B, T, self.num_kv_heads, self.d_head)).transpose(2, 3)
        v = self.v_proj(x).view(batch, B, T, self.num_kv_heads, self.d_head).transpose(2, 3)

        cos, sin = self._rope(T, x.device, x.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=-3)
            v = v.repeat_interleave(self.kv_repeat, dim=-3)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (batch, B, H, T, d_head)
        out = out.transpose(2, 3).contiguous().view(batch, B, T, self.num_heads * self.d_head)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

    def _attn_weights(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            The attention weights and output gate computed from (and frozen
            at) the actual layer input x0. Returns (attn, sigmoid(gate)).
        """
        batch, B, T, D = x0.shape
        q = self.q_proj(x0).view(batch, B, T, self.num_heads, self.d_head * 2)
        q, gate = torch.chunk(q, 2, dim=-1)
        gate = gate.reshape(batch, B, T, self.num_heads * self.d_head)
        q = self.q_norm(q).transpose(2, 3)
        k = self.k_norm(self.k_proj(x0).view(batch, B, T, self.num_kv_heads, self.d_head)).transpose(2, 3)
        cos, sin = self._rope(T, x0.device, x0.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=-3)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x0.device), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        return torch.softmax(scores, dim=-1), torch.sigmoid(gate)

    def frozen_forward(self, x: torch.Tensor, x0: torch.Tensor, affine: bool = True) -> torch.Tensor:
        """
            Frozen-routing linearization of the block, evaluated at x: the
            attention weights and the output gate are computed from (and
            frozen at) the actual layer input x0, so the map
            x -> frozen_forward(x, x0) is LINEAR across positions (through
            the value path) and equals forward(x0) at x = x0 exactly. Used
            by KnowledgeMatrixComputer(mixer_mode="frozen") and
            KnowledgeRowComputer. With affine=False the value/output
            projections are applied without their biases (the knowledge
            matrix column pass); biases flow through the affine=True pass.
        """
        batch, B, T, D = x.shape
        attn0, gate0 = self._attn_weights(x0)

        v_bias = self.v_proj.bias if (affine and self.v_proj.bias is not None) else None
        v = F.linear(x, self.v_proj.weight, v_bias)
        v = v.view(batch, B, T, self.num_kv_heads, self.d_head).transpose(2, 3)
        if self.kv_repeat > 1:
            v = v.repeat_interleave(self.kv_repeat, dim=-3)

        out = attn0 @ v  # (1,1,H,T,T) broadcast against (batch,B,H,T,d_head)
        out = out.transpose(2, 3).reshape(batch, B, T, self.num_heads * self.d_head)
        out = out * gate0
        o_bias = self.o_proj.bias if (affine and self.o_proj.bias is not None) else None
        return F.linear(out, self.o_proj.weight, o_bias)


class GatedDeltaNet(nn.Module):
    """
        Linear attention block of Qwen3-Next / Qwen3.5 (Gated DeltaNet):
        - fused q/k/v projection followed by a short depthwise causal
          convolution with SiLU,
        - per-head gated delta rule recurrence over the sequence
          (decay gate g from a/dt_bias/A_log, write strength beta from b),
        - RMS normalization gated by silu(z),
        - output projection.

        Matches the reference implementation in Hugging Face transformers
        (models/qwen3_5, Apache-2.0), using the recurrent (step-by-step)
        form of the delta rule. The block maps d_model -> d_model, so the
        knowledge matrix computation treats it like an activation
        (elementwise post/pre ratio), as it does for MultiHeadAttention.

        Args:
            d_model (int): The dimension of the model.
            num_v_heads (int): The number of value heads.
            num_k_heads (int): The number of key (and query) heads.
            head_k_dim (int): The dimension of each key/query head.
            head_v_dim (int): The dimension of each value head.
            conv_kernel_size (int): Kernel size of the causal convolution.
            rms_norm_eps (float): Epsilon of the gated RMS normalization.
    """
    def __init__(
            self,
            d_model: int,
            num_v_heads: int,
            num_k_heads: int,
            head_k_dim: int = 128,
            head_v_dim: int = 128,
            conv_kernel_size: int = 4,
            rms_norm_eps: float = 1e-6
        ) -> None:
        super().__init__()
        if num_v_heads % num_k_heads != 0:
            raise ValueError("num_v_heads must be divisible by num_k_heads.")

        self.d_model = d_model
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_kernel_size = conv_kernel_size

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            padding=conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))
        A = torch.empty(num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False)

        self.in_proj_qkv = nn.Linear(d_model, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d_model, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d_model, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(d_model, num_v_heads, bias=False)

    @staticmethod
    def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)

    def _conv_silu_qkv(self, h: torch.Tensor, pre_activation: bool = False) -> torch.Tensor:
        """
            Fused q/k/v projection + depthwise causal convolution (+ SiLU
            unless pre_activation). h: (N, T, d_model) -> (N, T, conv_dim).
        """
        T = h.shape[1]
        mixed_qkv = self.in_proj_qkv(h).transpose(1, 2)  # (N, conv_dim, T)
        mixed_qkv = F.conv1d(
            mixed_qkv,
            weight=self.conv1d.weight,
            bias=self.conv1d.bias,
            padding=self.conv_kernel_size - 1,
            groups=self.conv_dim,
        )[:, :, :T]
        if not pre_activation:
            mixed_qkv = F.silu(mixed_qkv)
        return mixed_qkv.transpose(1, 2)  # (N, T, conv_dim)

    def _delta_rule(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            cdt: torch.dtype
        ) -> torch.Tensor:
        """
            Gated delta rule, recurrent form. query/key: (Nq, T, H, d_k),
            value: (N, T, H, d_v), g/beta: (Nq, T, H); Nq may be 1 while N
            is a batch (broadcast, used by the frozen linearization).
            Returns (N, T, H, d_v) in dtype cdt.
        """
        T = value.shape[1]
        query = self._l2norm(query.to(cdt)).transpose(1, 2)
        key = self._l2norm(key.to(cdt)).transpose(1, 2)
        value = value.to(cdt).transpose(1, 2)
        beta = beta.to(cdt).transpose(1, 2)
        g = g.to(cdt).transpose(1, 2)
        query = query / math.sqrt(self.head_k_dim)

        N, H = value.shape[0], value.shape[1]
        state = torch.zeros(N, H, self.head_k_dim, self.head_v_dim, dtype=cdt, device=value.device)
        core_out = torch.zeros(N, H, T, self.head_v_dim, dtype=cdt, device=value.device)
        for t in range(T):
            q_t = query[:, :, t]
            k_t = key[:, :, t]
            v_t = value[:, :, t]
            g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, t].unsqueeze(-1)

            state = state * g_t
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_out[:, :, t] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

        return core_out.transpose(1, 2)  # (N, T, H, d_v)

    def _routing(self, h: torch.Tensor, cdt: torch.dtype, qkv: Union[torch.Tensor, None] = None):
        """
            The recurrence inputs computed from h: (N, T, d_model).
            Returns (query, key, g, beta) with the kv grouping expanded.
        """
        T = h.shape[1]
        if qkv is None:
            qkv = self._conv_silu_qkv(h)
        query = qkv[..., :self.key_dim]
        key = qkv[..., self.key_dim:2 * self.key_dim]
        query = query.reshape(-1, T, self.num_k_heads, self.head_k_dim)
        key = key.reshape(-1, T, self.num_k_heads, self.head_k_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        beta = self.in_proj_b(h).sigmoid()
        g = -self.A_log.to(cdt).exp() * F.softplus(self.in_proj_a(h).to(cdt) + self.dt_bias.to(cdt))
        return query, key, g, beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, B, T, D = x.shape
        # The reference computes the recurrence in float32; keep float64
        # when the network runs in float64 (this library's exactness tests).
        cdt = torch.float64 if x.dtype == torch.float64 else torch.float32
        h = x.reshape(batch * B, T, D)

        z = self.in_proj_z(h).reshape(batch * B, T, self.num_v_heads, self.head_v_dim)
        qkv = self._conv_silu_qkv(h)
        value = qkv[..., 2 * self.key_dim:].reshape(batch * B, T, self.num_v_heads, self.head_v_dim)
        query, key, g, beta = self._routing(h, cdt, qkv=qkv)

        core_out = self._delta_rule(query, key, value, g, beta, cdt).to(x.dtype)
        core_out = self.norm(core_out, z)
        core_out = core_out.reshape(batch, B, T, self.value_dim)
        return self.out_proj(core_out)

    def frozen_forward(self, x: torch.Tensor, x0: torch.Tensor, affine: bool = True) -> torch.Tensor:
        """
            Frozen-routing linearization of the block, evaluated at x: the
            recurrence inputs q, k and the gates g, beta, the conv-SiLU
            activation ratio of the value path, the gate silu(z) and the
            output RMS are all computed from (and frozen at) the actual
            layer input x0, so the map x -> frozen_forward(x, x0) is LINEAR
            across positions and channels, and equals forward(x0) at
            x = x0 exactly. Used by KnowledgeMatrixComputer(mixer_mode=
            "frozen") and KnowledgeRowComputer. All projections of this
            block are bias-free, so `affine` has no effect; it is accepted
            for interface uniformity with the other linearizable layers.
        """
        batch, B, T, D = x.shape
        cdt = torch.float64 if x.dtype == torch.float64 else torch.float32
        h = x.reshape(batch * B, T, D)
        h0 = x0.reshape(-1, T, D)

        # Frozen routing and gates from x0.
        pre0_full = self._conv_silu_qkv(h0, pre_activation=True)
        query0, key0, g0, beta0 = self._routing(h0, cdt, qkv=F.silu(pre0_full))
        z0 = self.in_proj_z(h0).reshape(-1, T, self.num_v_heads, self.head_v_dim)
        gate0 = F.silu(z0)

        # Value path of x, with the conv SiLU replaced by its (frozen)
        # activation ratio at x0 -- exact at x0, linear in x. At a
        # coordinate where the pre-activation vanishes the ratio is the
        # limit silu'(0) = 1/2 (the actual value there is 0 either way).
        pre0 = pre0_full[..., 2 * self.key_dim:]
        ratio0 = torch.where(pre0 == 0, torch.full_like(pre0, 0.5), F.silu(pre0) / pre0)
        pre = self._conv_silu_qkv(h, pre_activation=True)[..., 2 * self.key_dim:]
        value = (ratio0 * pre).reshape(batch * B, T, self.num_v_heads, self.head_v_dim)

        # Frozen recurrence (linear in the values), and the frozen output
        # normalization: RMS taken from the actual output at x0.
        core = self._delta_rule(query0, key0, value, g0, beta0, cdt).to(x.dtype)
        value0 = F.silu(pre0).reshape(-1, T, self.num_v_heads, self.head_v_dim)
        core0 = self._delta_rule(query0, key0, value0, g0, beta0, cdt).to(x.dtype)
        rms0 = torch.sqrt(torch.mean(core0 ** 2, dim=-1, keepdim=True) + self.norm.eps)

        core = (core / rms0) * self.norm.weight * gate0
        core = core.reshape(batch, B, T, self.value_dim)
        return self.out_proj(core)


class JumpReLU(nn.ReLU):
    """
        JumpReLU activation: z * 1[z > threshold], with per-feature thresholds.
    """
    def __init__(self, threshold: torch.Tensor):
        super().__init__()
        self.register_buffer("threshold", threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x > self.threshold).float()


class TopKActivation(nn.ReLU):
    """
        TopK activation: keeps only the top-k activations, zeros the rest.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = torch.topk(x, self.k, dim=-1)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_idx, topk_vals)
        return result


class PositionalEncoding(nn.Module):
    """
        Positional encoding for the transformer model.

        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input.
        
        Inspired by: https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
        which uses the positional encoding from the Attention is All You Need paper.
    """
    def __init__(self, d_model: int, max_len: int=5000) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even.")

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
        Multi-head attention for the transformer model. 

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of heads.
            mask (torch.Tensor): The mask to apply to the attention scores.
        
        Inspired by: https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
        which is inspired by the Attention is All You Need paper.
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: Union[int, None]=None,
            mask: Union[torch.Tensor, None]=None
        ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_model // num_heads
        self.kv_repeat = num_heads // num_kv_heads
        self.mask = mask

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, num_kv_heads * self.d_head)
        self.V = nn.Linear(d_model, num_kv_heads * self.d_head)
        self.O = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, B, T, D = x.shape

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        Q = Q.view(batch, B, T, self.num_heads, self.d_head).transpose(2, 3)
        K = K.view(batch, B, T, self.num_kv_heads, self.d_head).transpose(2, 3)
        V = V.view(batch, B, T, self.num_kv_heads, self.d_head).transpose(2, 3)

        if self.kv_repeat > 1:
            K = K.repeat_interleave(self.kv_repeat, dim=-3)
            V = V.repeat_interleave(self.kv_repeat, dim=-3)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(2, 3).contiguous().view(batch, B, T, D)
        return self.O(out)

    def _attn_weights(self, x0: torch.Tensor) -> torch.Tensor:
        """The softmax attention weights at the actual input x0."""
        batch, B, T, D = x0.shape
        Q = self.Q(x0).view(batch, B, T, self.num_heads, self.d_head).transpose(2, 3)
        K = self.K(x0).view(batch, B, T, self.num_kv_heads, self.d_head).transpose(2, 3)
        if self.kv_repeat > 1:
            K = K.repeat_interleave(self.kv_repeat, dim=-3)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float("-inf"))
        return torch.softmax(scores, dim=-1)

    def frozen_forward(self, x: torch.Tensor, x0: torch.Tensor, affine: bool = True) -> torch.Tensor:
        """
            Frozen-routing linearization of the block, evaluated at x: the
            softmax attention weights are computed from (and frozen at) the
            actual layer input x0, so the map x -> frozen_forward(x, x0) is
            LINEAR across positions (through the value path) and equals
            forward(x0) at x = x0 exactly. Used by
            KnowledgeMatrixComputer(mixer_mode="frozen") and
            KnowledgeRowComputer. With affine=False the value/output
            projections are applied without their biases (the knowledge
            matrix column pass); biases flow through the affine=True pass.
        """
        batch, B, T, D = x.shape
        attn0 = self._attn_weights(x0)

        v_bias = self.V.bias if (affine and self.V.bias is not None) else None
        v = F.linear(x, self.V.weight, v_bias)
        v = v.view(batch, B, T, self.num_kv_heads, self.d_head).transpose(2, 3)
        if self.kv_repeat > 1:
            v = v.repeat_interleave(self.kv_repeat, dim=-3)

        out = attn0 @ v  # (1,1,H,T,T) broadcast against (batch,B,H,T,d_head)
        out = out.transpose(2, 3).reshape(batch, B, T, self.num_heads * self.d_head)
        o_bias = self.O.bias if (affine and self.O.bias is not None) else None
        return F.linear(out, self.O.weight, o_bias)

    def eval(self) -> None:
        self.Q.eval()
        self.K.eval()
        self.V.eval()
        self.O.eval()

    def train(self) -> None:
        self.Q.train()
        self.K.train()
        self.V.train()
        self.O.train()

# Layers that the knowledge matrix computation treats as activations:
# the elementwise post/pre ratio of their saved activations is applied as
# a diagonal map. Any layer type added here is automatically handled by
# both NN.forward (save mode) and KnowledgeMatrixComputer.
ACTIVATION_LAYERS = (
    nn.ELU, nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU,
    nn.Mish, nn.Softmax, nn.CELU, nn.Hardsigmoid, nn.Hardswish, nn.PReLU,
    nn.ReLU6, nn.Softplus, MultiHeadAttention, GatedAttention, GatedDeltaNet,
    SwiGLU,
)

# Layers that additionally support the frozen-routing linearization
# (frozen_forward): with KnowledgeMatrixComputer(mixer_mode="frozen") or
# KnowledgeRowComputer, these are applied as linear maps with their
# routing (attention weights, recurrence gates, GLU gates) frozen at the
# actual input, instead of the elementwise post/pre ratio. The frozen map
# still reproduces the layer output exactly at the actual input, so the
# knowledge matrix row-sum invariant is preserved -- but attribution can
# flow ACROSS token positions through the value paths, which the ratio
# treatment (a diagonal map) cannot express.
LINEARIZABLE_LAYERS = (MultiHeadAttention, GatedAttention, GatedDeltaNet, SwiGLU)
