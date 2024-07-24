import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import Compose, RandomResizedCrop, RandomPerspective

from typing import List


class LPIPS(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        normalize: bool = True,
        pre_relu: bool = True,
        size: int = 224,
    ):
        """LPIPS (more semantic than direct L2 loss)
        reference: https://github.com/yael-vinker/CLIPasso/blob/main/models/loss.py
        """

        super().__init__()

        self.device = device

        self.normalize = normalize
        self.size = size

        augementations = [
            RandomPerspective(fill=0, p=1.0, distortion_scale=0.5),
            RandomResizedCrop(self.size, scale=(0.8, 0.8), ratio=(1.0, 1.0)),
        ]
        self.augment_trans = Compose(augementations)
        self.feature_extractor = FeatureExtractor(pre_relu).to(self.device)

        # mean instead of sum to avoid super high range if not normalized
        self.diff = torch.sum if self.normalize else torch.mean

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        return x / (norm + eps)

    @staticmethod
    def resize(x: torch.Tensor, size: int, mode="bilinear") -> torch.Tensor:
        return F.interpolate(x, size=size, mode=mode)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, train: bool = True):
        """Compare VGG features between inputs"""

        # get VGG features
        sketch_augs, img_augs = [self.resize(pred, self.size)], [
            self.resize(target, self.size)
        ]
        if train:
            for _ in range(4):
                pair = self.augment_trans(torch.cat([pred, target]))  # [2, C, H, W]
                sketch_augs.append(pair[0:1])  # [1, C, H, W]
                img_augs.append(pair[1:])  # [1, C, H, W]

        xs = torch.cat(sketch_augs, dim=0)
        ys = torch.cat(img_augs, dim=0)

        pred = self.feature_extractor(xs)
        target = self.feature_extractor(ys)

        # get L2 normalize features if needed
        if self.normalize:
            pred = [self._l2_normalize(f) for f in pred]
            target = [self._l2_normalize(f) for f in target]

        diffs = [self.diff((p - t) ** 2, 1) for (p, t) in zip(pred, target)]

        # spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        pre_relu: bool,
        weights: str = "DEFAULT",
        scale: List[float] = [0.485, 0.456, 0.406],
        shift: List[float] = [0.229, 0.224, 0.225],
    ):
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg16(weights=weights).features

        # set breakpoints to extract features
        self.breakpoints = [0, 4, 9, 16, 23, 30]
        if pre_relu:
            for i, _ in enumerate(self.breakpoints[1:]):
                self.breakpoints[i + 1] -= 1

        # split at the maxpools
        for i, b in enumerate(self.breakpoints[:-1]):
            ops = torch.nn.Sequential()
            for idx in range(b, self.breakpoints[i + 1]):
                op = vgg[idx]
                ops.add_module(str(idx), op)
            self.add_module(f"group{i}", ops)

        # gradients are not required
        for p in self.parameters():
            p.requires_grad = False

        # values to normalize inputs
        # referred to https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html
        self.register_buffer("shift", torch.Tensor(shift).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.Tensor(scale).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.shift) / self.scale
        n_breakpoints = len(self.breakpoints)
        feats = []
        for idx in range(n_breakpoints - 1):
            x = getattr(self, f"group{idx}")(x)
            feats.append(x)

        return feats
