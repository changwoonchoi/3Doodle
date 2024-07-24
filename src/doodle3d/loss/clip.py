import torch
import torch.nn as nn

import clip.model as model

import collections
from typing import Dict, Any


class CLIPVisualEncoder(nn.Module):
    """
    CLIP encoder to extract features
    reference: https://github.com/yael-vinker/CLIPasso/blob/main/models/loss.py
    """

    def __init__(self, model: model.CLIP):
        super().__init__()

        self.model = model
        self.featuremaps = None

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i)
            )

    def make_hook(self, name: str):
        def hook(output: torch.Tensor):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(1, 0, 2)
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        self.featuremaps = collections.OrderedDict()
        fc_features = self.model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return {"fc": fc_features, "conv": featuremaps}
