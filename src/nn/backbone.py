import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, version, out_levels=(5,), pretrained=False, out_feature_dim=128):
        """Initializes the ResNet backbone.

        Args:
            version (str): Version of the ResNet backbone.
            out_levels (tuple): Which stage outputs to return. Defaults to (5,) (i.e. the last stage).
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
        """
        super(ResNetBackbone, self).__init__()
        model_versions = {
            "18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }
        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]
        model = model_fn(weights=weights if pretrained else None)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(model.conv1, model.bn1, model.relu),
                nn.Sequential(model.maxpool, model.layer1),
                model.layer2,
                model.layer3,
                model.layer4,
            ]
        )
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        self.reduction_factor = 2**5

    def forward(self, x):
        features = [x] if self.out_levels[0] == 0 else []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i + 1 in self.out_levels:
                features.append(x)
        return features


class EfficientNetBackbone(nn.Module):
    def __init__(self, version, out_levels=(8,), pretrained=False, out_feature_dim=128):
        """Initializes the EfficientNet backbone.

        Args:
            version (str): Version of the EfficientNet backbone.
            out_levels (tuple): Which stage outputs to return. Defaults to (8,) (i.e. the last stage).
            pretrained (bool): Whether to use pretrained weights. Defaults to False.
        """
        super(EfficientNetBackbone, self).__init__()
        model_versions = {
            "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        }
        if version not in model_versions:
            raise NotImplementedError
        model_fn, weights = model_versions[version]
        # last block is discarded because it would be redundant with the pooling layer
        model = model_fn(weights=weights if pretrained else None).features[:-1]
        self.stages = nn.ModuleList([model[i] for i in range(len(model))])
        self.out_levels = out_levels
        self.out_channels = [3] if self.out_levels[0] == 0 else []
        for i in self.out_levels:
            stage = self.stages[i - 1]
            last_conv = [m for m in stage.modules() if isinstance(m, nn.Conv2d)][-1]
            self.out_channels.append(last_conv.out_channels)
        self.out_channels = tuple(self.out_channels)
        self.reduction_factor = 2**5
        self.out_feature_dim = out_feature_dim
        self.fc = nn.Linear(self.out_channels[-1], out_feature_dim)

    def forward(self, x):
        # 支持4D和5D输入
        if x.dim() == 5:
            B, E, C, H, W = x.shape
            x = x.view(B * E, C, H, W)
            for i, stage in enumerate(self.stages):
                x = stage(x)
            # 全局平均池化
            x = nn.functional.adaptive_avg_pool2d(x, 1)  # [B*E, feature_dim, 1, 1]
            x = x.view(B, E, -1)  # [B, E, feature_dim]
            x = self.fc(x)  # [B, E, out_feature_dim]
            return x
        elif x.dim() == 4:
            B, C, H, W = x.shape
            for i, stage in enumerate(self.stages):
                x = stage(x)
            x = nn.functional.adaptive_avg_pool2d(x, 1)  # [B, feature_dim, 1, 1]
            x = x.view(B, -1)  # [B, feature_dim]
            x = self.fc(x)  # [B, out_feature_dim]
            return x
        else:
            raise ValueError(f"EfficientNetBackbone only supports 4D or 5D input, got {x.dim()}D")
        return features
