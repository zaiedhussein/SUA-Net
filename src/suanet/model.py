from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


class SpeckleVarianceAttention(nn.Module):
    """Channel-spatial attention driven by local feature variance."""

    def __init__(
        self,
        channels: int,
        kernel: int = 7,
        reduction: int = 8,
        attention_mode: str = "joint",
        variance_mode: str = "channel_mean",
    ) -> None:
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be a positive odd integer")
        if attention_mode not in {"joint", "channel_only", "spatial_only"}:
            raise ValueError("attention_mode must be joint, channel_only, or spatial_only")
        if variance_mode not in {"channel_mean", "channel_preserving"}:
            raise ValueError("variance_mode must be channel_mean or channel_preserving")
        self.kernel = kernel
        self.attention_mode = attention_mode
        self.variance_mode = variance_mode
        hidden = max(channels // reduction, 4)
        self.ch_gate = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, channels),
                nn.Sigmoid(),
            )
            if attention_mode in {"joint", "channel_only"}
            else None
        )
        spatial_channels = 1 if variance_mode == "channel_mean" else channels
        self.sp_proj = (
            nn.Sequential(
                nn.Conv2d(
                    spatial_channels,
                    spatial_channels,
                    kernel_size=3,
                    padding=1,
                    groups=spatial_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(spatial_channels),
                nn.Sigmoid(),
            )
            if attention_mode in {"joint", "spatial_only"}
            else None
        )

    def local_variance(self, features: torch.Tensor) -> torch.Tensor:
        padding = self.kernel // 2
        if self.variance_mode == "channel_mean":
            first_moment = features.mean(dim=1, keepdim=True)
            second_moment = features.square().mean(dim=1, keepdim=True)
        else:
            first_moment = features
            second_moment = features.square()
        local_mean = F.avg_pool2d(first_moment, self.kernel, stride=1, padding=padding)
        local_square_mean = F.avg_pool2d(second_moment, self.kernel, stride=1, padding=padding)
        variance = (local_square_mean - local_mean.square()).clamp_min(0)
        maximum = variance.flatten(2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        return variance / (maximum + 1e-6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = features.shape
        output = features
        if self.ch_gate is not None:
            channel_weight = self.ch_gate(features).view(batch, channels, 1, 1)
            output = output * channel_weight
        if self.sp_proj is not None:
            output = output * self.sp_proj(self.local_variance(features))
        return output


class DualPathLightweightAggregation(nn.Module):
    """Gated fusion of pointwise and depthwise-dilated feature paths."""

    def __init__(self, channels: int, dilation: int = 3) -> None:
        super().__init__()
        self.path_a = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.path_b = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pointwise = self.path_a(features)
        contextual = self.path_b(features)
        gate = self.gate(torch.cat([pointwise, contextual], dim=1))
        fused = gate * pointwise + (1.0 - gate) * contextual
        return self.norm(fused) + features


class MultiGranularityPooling(nn.Module):
    """Configurable global statistics followed by a residual MLP."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.4,
        statistics: tuple[str, ...] = ("avg", "max", "std"),
    ) -> None:
        super().__init__()
        valid_statistics = {"avg", "max", "std", "skew"}
        if not statistics or len(set(statistics)) != len(statistics):
            raise ValueError("statistics must contain one or more unique entries")
        if not set(statistics) <= valid_statistics:
            raise ValueError("statistics entries must be avg, max, std, or skew")
        self.statistics = tuple(statistics)
        self.proj = nn.Sequential(
            nn.Linear(in_features * len(self.statistics), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def pool(
        features: torch.Tensor,
        statistics: tuple[str, ...] = ("avg", "max", "std"),
    ) -> torch.Tensor:
        flattened = features.flatten(2)
        average = flattened.mean(dim=-1)
        pooled = []
        for statistic in statistics:
            if statistic == "avg":
                pooled.append(average)
            elif statistic == "max":
                pooled.append(flattened.amax(dim=-1))
            elif statistic == "std":
                pooled.append(flattened.std(dim=-1, correction=0))
            elif statistic == "skew":
                centered = flattened - average.unsqueeze(-1)
                scale = flattened.std(dim=-1, correction=0).clamp_min(1e-6)
                pooled.append(centered.pow(3).mean(dim=-1) / scale.pow(3))
            else:
                raise ValueError(f"Unsupported pooling statistic: {statistic}")
        return torch.cat(pooled, dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(self.pool(features, self.statistics))
        hidden = hidden + self.residual_block(hidden)
        return self.classifier(hidden)


class StandardGAPHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(features))


class ScratchEncoder(nn.Module):
    """Small fallback encoder used for unit tests and CPU smoke tests."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    in_ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=in_ch,
                    bias=False,
                ),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.stages = nn.Sequential(
            block(32, 64, 1),
            block(64, 128, 2),
            block(128, 128, 1),
            block(128, 256, 2),
            block(256, 256, 1),
            block(256, 512, 2),
            block(512, 512, 1),
        )
        self.out_channels = 512

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(inputs))


class SUANet(nn.Module):
    """Configurable SUA-Net used by all training and evaluation commands."""

    def __init__(
        self,
        encoder_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.4,
        freeze_blocks: int = 2,
        sva_kernel: int = 7,
        sva_reduction: int = 8,
        dla_dilation: int = 3,
        use_sva: bool = True,
        use_dla: bool = True,
        use_mgp: bool = True,
        sva_attention_mode: str = "joint",
        sva_variance_mode: str = "channel_mean",
        mgp_statistics: tuple[str, ...] = ("avg", "max", "std"),
        probe_size: int = 224,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.use_sva = use_sva
        self.use_dla = use_dla
        self.use_mgp = use_mgp
        self.mgp_statistics = tuple(mgp_statistics)

        if encoder_name.lower() in {"scratch", "__scratch__"}:
            self.backbone = ScratchEncoder()
            feature_channels = self.backbone.out_channels
        else:
            if timm is None:
                raise ImportError(
                    "timm is required for pretrained backbones. "
                    "Install the project with `pip install -e .`."
                )
            self.backbone = timm.create_model(
                encoder_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
            )
            feature_channels = self._infer_feature_channels(probe_size)
            self._freeze_backbone_blocks(freeze_blocks)

        self.feature_channels = feature_channels
        self.dla = (
            DualPathLightweightAggregation(feature_channels, dilation=dla_dilation)
            if use_dla
            else nn.Identity()
        )
        self.sva = (
            SpeckleVarianceAttention(
                feature_channels,
                kernel=sva_kernel,
                reduction=sva_reduction,
                attention_mode=sva_attention_mode,
                variance_mode=sva_variance_mode,
            )
            if use_sva
            else nn.Identity()
        )
        self.head = (
            MultiGranularityPooling(
                feature_channels,
                hidden_dim=max(feature_channels // 2, 256),
                num_classes=num_classes,
                dropout=dropout,
                statistics=self.mgp_statistics,
            )
            if use_mgp
            else StandardGAPHead(feature_channels, num_classes, dropout)
        )

    def _forward_backbone(self, inputs: torch.Tensor):
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(inputs)
        return self.backbone(inputs)

    def _infer_feature_channels(self, probe_size: int) -> int:
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            output = self._forward_backbone(torch.zeros(1, 3, probe_size, probe_size))
        self.backbone.train(was_training)
        if isinstance(output, (tuple, list)):
            output = output[-1]
        if output.ndim not in {3, 4}:
            raise RuntimeError(
                f"Backbone {self.encoder_name} returned unsupported shape {tuple(output.shape)}"
            )
        declared = int(getattr(self.backbone, "num_features", 0))
        if output.ndim == 3:
            return int(output.shape[-1])
        if declared and output.shape[-1] == declared:
            return declared
        return int(output.shape[1])

    def _as_feature_map(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim == 4:
            if output.shape[1] == self.feature_channels:
                return output
            if output.shape[-1] == self.feature_channels:
                return output.permute(0, 3, 1, 2).contiguous()
        if output.ndim == 3 and output.shape[-1] == self.feature_channels:
            token_count = output.shape[1]
            side = int(token_count**0.5)
            if side * side != token_count:
                token_count -= 1
                side = int(token_count**0.5)
                if side * side != token_count:
                    raise RuntimeError(
                        f"Cannot reshape {output.shape[1]} backbone tokens into a feature map"
                    )
                output = output[:, 1:, :]
            return output.transpose(1, 2).reshape(
                output.shape[0], self.feature_channels, side, side
            )
        raise RuntimeError(
            f"Backbone {self.encoder_name} returned shape {tuple(output.shape)}; "
            "a spatial feature map could not be derived"
        )

    def _freeze_backbone_blocks(self, count: int) -> None:
        if count <= 0 or not hasattr(self.backbone, "blocks"):
            return
        for index, block in enumerate(self.backbone.blocks):
            if index >= count:
                break
            for parameter in block.parameters():
                parameter.requires_grad = False

    def extract_backbone_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self._forward_backbone(inputs)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        return self._as_feature_map(features)

    def extract_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.extract_backbone_feature_map(inputs)
        features = self.dla(features)
        return self.sva(features)

    def extract_pooled_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return MultiGranularityPooling.pool(
            self.extract_feature_map(inputs), self.mgp_statistics
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_feature_map(inputs))

    def enable_mc_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def target_layer(self) -> nn.Module:
        if self.use_sva:
            return self.sva
        if self.use_dla:
            return self.dla
        return self.backbone

    def parameter_counts(self) -> dict[str, int]:
        return {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in self.parameters()),
        }


def build_model(model_config, *, pretrained: bool | None = None) -> SUANet:
    kwargs = {
        "encoder_name": model_config.encoder_name,
        "pretrained": model_config.pretrained if pretrained is None else pretrained,
        "num_classes": model_config.num_classes,
        "dropout": model_config.dropout,
        "freeze_blocks": model_config.freeze_blocks,
        "sva_kernel": model_config.sva_kernel,
        "sva_reduction": model_config.sva_reduction,
        "dla_dilation": model_config.dla_dilation,
        "use_sva": model_config.use_sva,
        "use_dla": model_config.use_dla,
        "use_mgp": model_config.use_mgp,
        "sva_attention_mode": model_config.sva_attention_mode,
        "sva_variance_mode": model_config.sva_variance_mode,
        "mgp_statistics": tuple(model_config.mgp_statistics),
    }
    return SUANet(**kwargs)
