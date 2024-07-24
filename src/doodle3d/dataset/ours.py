import torch
import torch.nn.functional as F
import numpy as np

import os
import json
from tqdm import tqdm
from typing import Tuple, Union

from doodle3d.dataset.base import DataSet


class Synthetic(DataSet):
    def __init__(
        self,
        projection: str = "PERSP",
        **kwargs,  # blender_coord = True
    ):
        """DataLoader for our synthetic scenes"""

        super().__init__(**kwargs)

        assert projection in ["ORTHO", "PERSP"]
        assert self.mode in ["train", "val", "test"]

        self.projection = projection
        self.basedir = f"{self.root}/{self.scene}"
        with open(f"{self.basedir}/{self.mode}/transform.json", "r") as f:
            self.info = json.load(f)
        self._len = len(self.info["frames"])
        if self._num_imgs < self._len and self.mode != "test":
            self.info["frames"] = self.info["frames"][: self._num_imgs]
            self._len = self._num_imgs

        raw_size = [self.info["height"], self.info["width"]]
        self.H, self.W = raw_size if self.resize is None else self.resize
        if self.projection == "PERSP":
            self.ortho_scale = None  # not required
            self.F = 0.5 * self.W / np.tan(self.info["camera_angle_x"] / 2)
        else:
            self.ortho_scale = self.info["ortho_scale"]  # left - right
            self.F = torch.inf  # not required but set as infinite

        self.preprocess()

    def __len__(self):
        return self._len

    @property
    def get_HWF(self) -> Tuple[Union[int, float]]:
        return self.H, self.W, self.F

    @property
    def get_intrinsic(self) -> torch.Tensor:
        mat = (
            torch.Tensor([[self.F, 0, self.W / 2], [0, self.F, self.H / 2], [0, 0, 1]])
            .float()
            .to(self.device)
        )

        return mat

    def preprocess(self) -> None:
        data = {"rgbs": [], "rays": [], "poses": []}

        mode = "eval" if self.mode == "val" else self.mode
        pbar = tqdm(self.info["frames"], desc=f"[{mode}] Loading data...")
        for frame in pbar:
            fname = os.path.join(self.basedir, self.mode, frame["file_path"])
            pose = np.array(frame["transform_matrix"]).astype(np.float32)
            pose = torch.from_numpy(pose)
            data["rgbs"].append(self.load_image(fname))
            data["rays"].append(self.convert_rays(pose))
            data["poses"].append(pose)

        data = self.stack(data)

        data["masks"] = data["rgbs"][..., -1]
        if self.white_bkgd:
            data["rgbs"] = data["rgbs"][..., :-1] * data["rgbs"][..., -1:] + (
                1.0 - data["rgbs"][..., -1:]
            )
        else:
            data["rgbs"] = data["rgbs"][..., :-1] * data["rgbs"][..., -1:]

        self.rgbs = data["rgbs"]  # [n, H, W, 3]
        self.masks = data["masks"]  # [n, H, W]
        self.poses = data["poses"]  # [n, 4, 4]
        self.rays = data["rays"]  # [n, H, W, 6]
