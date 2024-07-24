import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

import clip.clip as clip
from robust_loss_pytorch.general import lossfun as robustfn

from typing import List, Dict, Any

from doodle3d.loss.clip import CLIPVisualEncoder
from doodle3d.loss.lpips import LPIPS
from doodle3d.utils.misc import MODEL_TYPES, DIST_TYPES, get_augment_trans
from doodle3d.utils.math_utils import cos_distance


class Loss(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        contour: Dict[str, Dict[str, Any]] = None,
        curve: Dict[str, Dict[str, Any]] = None,
    ):
        super().__init__()

        self.device = device

        loss_types = {
            "direct": CLIPLoss,
            "conv": CLIPConvLoss,
            "joint": JointLoss,
        }

        self.contour = []
        for key, params in contour.items():
            assert key in loss_types.keys(), "Invalid type of loss."
            self.contour.append(loss_types[key](self.device, **params))

        self.curve = []
        for key, params in curve.items():
            assert key in loss_types.keys(), "Invalid type of loss."
            self.curve.append(loss_types[key](self.device, **params))

    def forward(
        self,
        sketch: torch.Tensor,
        target: torch.Tensor,
        train: bool = True,
        only_sq: bool = False,
    ) -> torch.Tensor:
        """Compute overall loss.

        Args:
            sketch (torch.Tensor):  [1, H, W, C]
            target (torch.Tensor):  [1, H, W, C]
            train (bool, optional): Defaults to False.
        """

        sketch = sketch.permute(0, 3, 1, 2).to(self.device)  # [1, C, H, W]
        target = target.permute(0, 3, 1, 2).to(self.device)  # [1, C, H, W]

        loss_list = self.contour if only_sq else self.curve

        loss = 0
        for unit in loss_list:
            loss += unit(sketch, target, train=train)

        return loss


class CLIPLoss(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        model_type: str = "Vit/B-32",
        num_augs: int = 4,
        affine: bool = True,
    ):
        super().__init__()

        assert model_type in MODEL_TYPES

        self.device = device
        self.model, preprocess = clip.load(model_type, self.device, jit=False)
        self.model.eval()
        self.preprocess = Compose([preprocess.transforms[-1]])

        self.num_augs = num_augs
        self.augment_trans = get_augment_trans(affine=affine)

    def forward(
        self, sketch: torch.Tensor, target: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        """
        Args:
            sketch (torch.Tensor):  [1, C, H, W]
            target (torch.Tensor):  [1, C, H, W]
            train (bool, optional): Defaults to False.
        """

        target_ = self.preprocess(target).to(self.device)
        target_features = self.model.encode_image(target_).detach()

        if not train:  # eval
            with torch.no_grad():
                sketch = self.preprocess(sketch).to(self.device)
                sketch_features = self.model.encode_image(sketch)
                return cos_distance(sketch_features, target_features)

        sketch_augs = []
        for _ in range(self.num_augs):
            augmented_pair = self.augment_trans(torch.cat([sketch, target]))
            sketch_augs.append(augmented_pair[0].unsqueeze(0))

        sketch_batch = torch.cat(sketch_augs)
        sketch_features = self.model.encode_image(sketch_batch)

        loss = 0
        for i in range(self.num_augs):
            cos_dist = cos_distance(sketch_features[i : i + 1], target_features, dim=1)
            loss += cos_dist

        return loss


class CLIPConvLoss(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        model_type: str = "RN101",
        conv_loss_type: str = "L2",
        fc_loss_type: str = "Cos",
        num_augs: int = 4,
        affine: bool = True,
        conv_weights: List[float] = [0.0, 0.0, 1.0, 1.0, 0.0],
        c_weight: float = 1.0,
        fc_weight: float = 50.0,
    ):
        super().__init__()

        assert (conv_loss_type in DIST_TYPES[:3]) and (fc_loss_type in DIST_TYPES[:3])
        assert model_type in MODEL_TYPES

        self.device = device
        self.model_type = model_type
        self.model, preprocess = clip.load(self.model_type, self.device, jit=False)

        self._loss_metric = {
            "L1": self._L1,
            "L2": self._L2,
            "Cos": self._cosine,
        }

        self.conv_loss = self._loss_metric[conv_loss_type]
        self.fc_loss = self._loss_metric[fc_loss_type]

        if self.model_type.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model)
        else:
            self.visual_model = self.model.visual
            layers = list(self.visual_model.children())
            self.init_layers = torch.nn.Sequential(*layers)[:8]
            self.layers = layers[8:12]
            self.attn_pool2d = layers[12]

        self.model.eval()

        self.normalize = Compose(
            [
                preprocess.transforms[0],  # resize
                preprocess.transforms[1],  # centercrop
                preprocess.transforms[-1],  # normalize
            ]
        )

        self.num_augs = num_augs
        self.augment_trans = get_augment_trans(affine=affine)

        self.conv_weights = conv_weights
        self.c_weight = c_weight
        self.fc_weight = fc_weight

    @staticmethod
    def _L1(
        features1: List[torch.Tensor], featrues2: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        return [torch.abs(f1 - f2).mean() for f1, f2 in zip(features1, featrues2)]

    @staticmethod
    def _L2(
        features1: List[torch.Tensor], featrues2: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        return [torch.square(f1 - f2).mean() for f1, f2 in zip(features1, featrues2)]

    @staticmethod
    def _cosine(
        features1: List[torch.Tensor], features2: List[torch.Tensor], RN: bool = True
    ) -> List[torch.Tensor]:
        if RN:
            return [
                torch.square(f1, f2, dim=1).mean()
                for f1, f2 in zip(features1, features2)
            ]
        else:
            return [
                cos_distance(f1, f2, dim=1).mean()
                for f1, f2 in zip(features1, features2)
            ]

    def forward(
        self, sketch: torch.Tensor, target: torch.Tensor, train: bool = False
    ) -> torch.Tensor:
        """
        Args:
            sketch (torch.Tensor):  [1, C, H, W]
            target (torch.Tensor):  [1, C, H, W]
            train (bool, optional): Defaults to False.
        """

        sketch_augs = [self.normalize(sketch)]
        target_augs = [self.normalize(target)]

        if train:
            for _ in range(self.num_augs):
                pair = self.augment_trans(torch.cat([sketch, target]))
                sketch_augs.append(pair[0].unsqueeze(0))
                target_augs.append(pair[1].unsqueeze(0))

        sketches = torch.cat(sketch_augs, dim=0).to(self.device)
        targets = torch.cat(target_augs, dim=0).to(self.device)

        if self.model_type.startswith("RN"):
            skecth_features = self.encode_RN(sketches.contiguous())
            target_features = self.encode_RN(targets.detach())
        else:
            skecth_features = self.encode_ViT(sketches)
            target_features = self.encode_ViT(targets)

        conv_loss = 0
        conv_loss_units = self.conv_loss(
            skecth_features["conv"], target_features["conv"]
        )
        for w, value in zip(self.conv_weights, conv_loss_units):
            conv_loss += w * value
        fc_loss = cos_distance(
            skecth_features["fc"], target_features["fc"], dim=1
        ).mean()

        loss = conv_loss * self.c_weight + fc_loss * self.fc_weight

        return loss

    def encode_RN(self, x: torch.Tensor) -> Dict[str, Any]:
        def stem(m, x):
            x = m.relu1(m.bn1(m.conv1(x)))
            x = m.relu2(m.bn2(m.conv2(x)))
            x = m.relu3(m.bn3(m.conv3(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        featuremaps = [x]
        for layer in self.layers:
            x = layer(featuremaps[-1])
            featuremaps.append(x)
        y = self.attn_pool2d(x)
        features = {"fc": y, "conv": featuremaps}

        return features

    def encode_ViT(self, x: torch.Tensor) -> Dict[str, Any]:
        features = self.visual_encoder(x)

        return features


class JointLoss(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        loss_type: str = "LPIPS",
        size: int = 224,
        weight: float = 1.0,
        robust: bool = False,
        alpha: float = 1.0,
        scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.device = device
        self.loss_type = loss_type
        self.weight = weight

        self._lpips = LPIPS(self.device, size=size)
        self._loss_metric = {
            "L1": F.l1_loss,
            "L2": F.mse_loss,
            "Cos": cos_distance,
            "LPIPS": self._lpips_diff,
        }
        self.loss_fn = self._loss_metric[self.loss_type]

        self.robust = robust
        self.alpha = torch.Tensor([alpha]).to(self.device)
        self.scale = torch.Tensor([scale]).to(self.device)

    def _lpips_diff(self, x: torch.Tensor, y: torch.Tensor, train: bool = True):
        return self._lpips(x, y, train)

    def forward(
        self,
        sketch: torch.Tensor,
        target: torch.Tensor,
        train: bool = True,
    ) -> torch.Tensor:
        if self.loss_type == "LPIPS":
            loss = self.loss_fn(sketch, target, train)
            loss = (
                robustfn(loss, self.alpha, self.scale).mean()
                if self.robust
                else loss.mean()
            )
        else:
            loss = self.loss_fn(sketch, target)

        return self.weight * loss
