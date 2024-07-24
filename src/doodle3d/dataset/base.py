
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

import open3d as o3d

import os
import cv2
import random
import numpy as np
from abc import *
from PIL import Image
from functools import cached_property
from typing import List, Dict, Union, Tuple, Any

from doodle3d.utils.misc import SFM_TYPES


class DataSet(metaclass=ABCMeta):
    def __init__(
        self,
        mode: str,
        root: str,
        scene: str,
        num_imgs: int = 100,
        near: float = 2.0,
        far: float = 6.0,
        white_bkgd: bool = True,
        resize: List[int] = [400, 400],
        init_points: bool = False,
        sfm_method: str = "sfm",
        sfm_params: Dict[str, Any] = None,
        device: str = "cuda:0",
        calc_gpu: bool = True,
        alc_gpu: bool = True,
    ):
        """Abstract class to manipulate various types of data"""
        # basic attributes
        self.device = device
        self.root = root
        self.scene = scene
        self.mode = mode
        self.white_bkgd = white_bkgd
        self.resize = resize

        self.alc_gpu = alc_gpu
        self.calc_device = self.device if calc_gpu else "cpu"

        self.near = near
        self.far = far

        # attributes related to SfM point cloud
        self.sfm_voxel_size: float = sfm_params["voxel_size"]
        self.sfm_filter_fn: str = sfm_params["filter_fn"]
        self.sfm_filter_params: List[Union[int, float]] = sfm_params["filter_params"]
        self.sfm_second_filter: bool = sfm_params["filter_again"]
        self.sfm_filter_ratio: float = sfm_params["filter_ratio"]
        assert len(self.sfm_filter_params) == 2

        # hidden attributes
        self._num_imgs = num_imgs
        self._indices = None
        self._count = 0

        # intialize properties as None
        self._len = 0

        self.rgbs: torch.FloatTensor = None
        self.masks: torch.FloatTensor = None
        self.poses: torch.FloatTensor = None
        self.rays: torch.FloatTensor = None

        # get extracted points or linemaps to initialize
        self.points = None

        assert sfm_method in SFM_TYPES
        self.sfm_method = sfm_method
        if sfm_method == "sfm":
            points_path = os.path.join(self.root, self.scene, "points/all.npy")
            self.init_points = init_points and os.path.exists(points_path)
            self.points = np.load(points_path) if self.init_points else None

    @abstractmethod
    def preprocess(self) -> None:
        """Preprocess dataset"""

    @property
    @abstractmethod
    def get_HWF(self) -> Tuple[Union[int, float]]:
        """Get image size and focal"""

    @property
    @abstractmethod
    def get_intrinsic(self) -> torch.Tensor:
        """Get camera intrinsic matrix"""

    @cached_property
    def transform(self):
        if self.resize is None:
            transform = Compose([ToTensor()])
        else:
            transform = Compose([Resize(tuple(self.resize)), ToTensor()])

        return transform

    @property
    def all_items(self) -> Dict[str, torch.Tensor]:
        """Return all items"""

        items = {
            "rgbs": self.rgbs,
            "rays": self.rays,
            "poses": self.poses,
        }

        masks = self.masks > 0.5
        masks = masks.float()
        items["masks"] = masks

        for k, v in items.items():
            items[k] = v.to(self.device)

        return items

    def stack(self, data: Dict[str, List]) -> Dict[str, torch.Tensor]:
        device = self.device if self.alc_gpu else "cpu"
        _data = {k: torch.stack(v).to(device) for k, v in data.items()}

        return _data

    def __getitem__(self, idx: int) -> Dict[str, torch.FloatTensor]:
        if self._len == 0:
            raise ValueError("Not completely computed yet.")

        idx = idx % self._len

        units = {
            "rgbs": self.rgbs[idx],
            "rays": self.rays[idx],
            "poses": self.poses[idx],
            "masks": (self.masks[idx] > 0.5).float(),
        }
        if not self.alc_gpu:
            units = {k: v.to(self.device) for k, v in units.items()}

        return units

    def get_dataloader(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = False,
        worker_init: bool = True,
    ) -> DataLoader:
        def seed_worker(_worker_id: int):
            worker_seed = torch.initial_seed() % (2**32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker if worker_init else None,
        )

        return loader

    def load_image(self, fname: str) -> torch.Tensor:
        img = self.transform(Image.open(fname))  # [C, H, W]
        img = img.permute(1, 2, 0).to(self.calc_device)  # [H, W, C]

        return img

    def set_rand_indices(self):
        if self._len == 0:
            raise ValueError("Not completely computed yet.")
        self._indices = torch.randperm(self._len)
        self._count = 0

    def next(self, batch_size: int = 1) -> Dict[str, torch.FloatTensor]:
        assert self._len % batch_size == 0

        if (self._indices is None) or (self._count >= self._len):
            self.set_rand_indices()

        idx = self._indices[self._count : self._count + batch_size]
        items = self[idx]
        self._count += batch_size

        return items

    def convert_rays(self, pose: torch.FloatTensor) -> torch.FloatTensor:
        pose = pose.to(self.calc_device)
        w_range, h_range = torch.arange(self.W, device=self.calc_device), torch.arange(
            self.H, device=self.calc_device
        )
        xs, ys = torch.meshgrid(w_range, h_range, indexing="xy")

        # blender projection
        dirs = torch.stack(
            [
                (xs - self.W / 2) / self.F,
                -(ys - self.H / 2) / self.F,
                -torch.ones_like(xs, device=self.calc_device),
            ],
            dim=-1,
        )
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)  # [H, W, 3]
        rays_o = pose[:3, -1].expand(rays_d.shape)  # [H, W, 3]

        return torch.concat([rays_o, rays_d], dim=-1)  # [H, W, 6]

    def fps_from_sfm(self, num_points: int) -> Tuple[Union[torch.Tensor, int]]:
        """FPS from points in sfm point cloud."""

        if (not self.init_points) or (self.sfm_method != "sfm"):
            return None, num_points

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # remove outliers
        sampled_pcd = pcd.voxel_down_sample(voxel_size=self.sfm_voxel_size)
        p1, p2 = self.sfm_filter_params
        if self.sfm_filter_fn == "statistic":
            cl, _ = sampled_pcd.remove_statistical_outlier(
                nb_neighbors=p1, std_ratio=p2
            )
        elif self.sfm_filter_fn == "radius":
            cl, _ = sampled_pcd.remove_radius_outlier(nb_points=p1, radius=p2)
        else:  # fn must be in ["statistic", "radius"]
            raise ValueError
        cl_pts = np.asarray(cl.points)

        # second filtering
        if self.sfm_second_filter:
            lower, upper = (
                self.sfm_filter_ratio * 100,
                (1 - self.sfm_filter_ratio) * 100,
            )
            perc_low = np.percentile(cl_pts, lower, axis=0, keepdims=True)
            perc_high = np.percentile(cl_pts, upper, axis=0, keepdims=True)
            low_filter = cl_pts > perc_low
            low_filter = np.logical_and(
                np.logical_and(low_filter[:, 0], low_filter[:, 1]), low_filter[:, 2]
            )
            high_filter = cl_pts < perc_high
            high_filter = np.logical_and(
                np.logical_and(high_filter[:, 0], high_filter[:, 1]), high_filter[:, 2]
            )
            filter = np.logical_and(low_filter, high_filter)
            cl.points = o3d.utility.Vector3dVector(cl_pts[filter])

        # apply fps to get initial coordinates of points
        n_all = len(cl.points)
        if n_all > num_points:
            computed = cl.farthest_point_down_sample(num_points)
            points = computed.points
        else:
            raise ValueError("Not enough points")
            # num_points = n_all
            # points = cl_filtered.points
        sampled = torch.from_numpy(np.asarray(points)).float().to(self.device)

        return sampled, num_points

    def farthest_sampling(self, num_samples: int) -> Tuple[Union[torch.Tensor, int]]:
        return self.fps_from_sfm(num_samples)