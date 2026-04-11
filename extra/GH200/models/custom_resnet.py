"""
    Parameterized ResNet builder for GH200 benchmarking.

    Generalizes the ResNet-18 pattern from knowledgematrix/models/resnet18.py
    to support arbitrary depth (via blocks_per_stage) and width (via base_width).
"""
from knowledgematrix.neural_net import NN


class CustomResNet(NN):
    """
        A parameterized residual network for benchmarking knowledge matrix computation.

        The network follows the standard ResNet basic-block pattern:
        each block is conv3x3 → batchnorm → relu → conv3x3 → batchnorm + skip connection.

        Channel widths double at each stage: base_width, base_width*2, base_width*4, base_width*8.
        The first block of stages 2+ uses stride=2 for spatial downsampling.

        Args:
            input_shape (tuple[int]): Shape of the input (C, H, W).
            num_classes (int): Number of output classes.
            base_width (int): Number of channels in the first stage.
            blocks_per_stage (list[int]): Number of residual blocks per stage.
                E.g., [2,2,2,2] gives ResNet-18 depth, [3,4,6,3] gives ResNet-34 depth.
            save (bool): Whether to save activations for knowledge matrix computation.
            device (str): Device to place the model on.
    """

    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            base_width: int,
            blocks_per_stage: list[int],
            save: bool = False,
            device: str = "cpu"
    ) -> None:
        super().__init__(input_shape, save, device)

        widths = [base_width * (2 ** i) for i in range(len(blocks_per_stage))]

        # Stem
        self.conv(input_shape[0], widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(widths[0])
        self.relu()
        if input_shape[1] >= 64:
            self.maxpool(kernel_size=3, stride=2, padding=1)

        # Stages
        in_channels = widths[0]
        for stage_idx, (num_blocks, out_channels) in enumerate(zip(blocks_per_stage, widths)):
            for block_idx in range(num_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                self._make_basic_block(in_channels, out_channels, stride)
                in_channels = out_channels

        # Head
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_features=widths[-1], out_features=num_classes)

    def _make_basic_block(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Build a single residual basic block: conv→bn→relu→conv→bn + skip."""
        start_skip = self.get_num_layers()
        self.conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchnorm(out_channels)
        self.relu()
        self.conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm(out_channels)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
