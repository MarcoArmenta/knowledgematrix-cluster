"""
    Benchmark configuration: model ladder, sweep parameters, and result schema constants.
"""
from dataclasses import dataclass, field


@dataclass
class ResNetConfig:
    """A single ResNet configuration in the model ladder."""
    base_width: int
    blocks_per_stage: list[int] = field(default_factory=list)

    @property
    def depth_label(self) -> int:
        """Total number of layers (stem conv + 2 convs per block + final linear)."""
        return 2 + sum(2 * b for b in self.blocks_per_stage)

    @property
    def label(self) -> str:
        """Human-readable label, e.g. 'R18-w128'."""
        return f"R{self.depth_label}-w{self.base_width}"


MODEL_LADDER: list[ResNetConfig] = [
    ResNetConfig(128, [2, 2, 2, 2]),   # R18-w128
    ResNetConfig(256, [2, 2, 2, 2]),   # R18-w256
    ResNetConfig(128, [3, 4, 6, 3]),   # R34-w128
    ResNetConfig(256, [3, 4, 6, 3]),   # R34-w256
    ResNetConfig(128, [4, 6, 8, 4]),   # R46-w128
    ResNetConfig(256, [4, 6, 8, 4]),   # R46-w256
]

INPUT_SIZES: list[tuple[int, int, int]] = [
    (3, 32, 32),
    (3, 64, 64),
    (3, 128, 128),
]

COLUMN_BATCH_SIZES: list[int] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

NUM_SAMPLES: list[int] = [10, 100, 1000, 10000]

NUM_CLASSES_LIST: list[int] = [10, 100, 1000]

ALLOCATORS: list[str] = ["default", "rmm"]

WARMUP_SAMPLES: int = 2
