import torch
import torch.nn as nn
import torch.optim as optim

import pydiffvg

import random
from typing import Tuple, Dict, Any, List

from doodle3d.dataset.base import DataSet
from doodle3d.utils.misc import (
    OPTIM_TYPES,
    PROJ_TYPES,
    RAND_TYPES,
    get_rand_fn,
    get_mean_dist,
    blender2world,
    rand_on_line,
    HWF,
)
from doodle3d.utils.math_utils import euclidean_distance


class CurveRenderer(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        proj_mode: str = "ortho",
        proj_scale: float = 1.0,
        blender: bool = True,
        use_sfm: bool = True,
        gap: float = 0.01,
        center: List[float] = [0.0, 0.0, 0.0],
        rand_mode: str = "hemisphere",
        upside: bool = True,
        boundaries: List[float] = [0.2],
        stroke_width: float = 1.5,
        num_strokes: int = 16,
        num_segments: int = 4,
        pts_per_seg: int = 4,
        eps: float = 0.01,
        optim_type: str = "Default",
        num_stages: int = 1,
        add_noise: bool = False,
        noise_thres: float = 0.5,
        color_thres: float = 0.0,
        pts_dist_thres: float = 0.015,
        **kwargs,
    ):
        """Module to compute view-independent lines"""

        super().__init__()

        self.device = device

        assert proj_mode in PROJ_TYPES
        self.proj_mode = proj_mode
        self.projection = (
            self.perspective if self.proj_mode == "persp" else self.orthographic
        )
        self.proj_scale = proj_scale
        self.blender = blender

        self.use_sfm = use_sfm
        self.start_points: torch.Tensor = None

        self.gap = gap
        self.center = torch.Tensor(center).to(self.device)

        assert rand_mode in RAND_TYPES
        self.set_random_point = get_rand_fn(rand_mode == "hemisphere", z_nonneg=upside)
        self.boundaries = boundaries[0] if rand_mode == "hemisphere" else boundaries

        self.stroke_width = stroke_width

        self.num_strokes = num_strokes
        self.num_segments = num_segments
        self.pts_per_seg = pts_per_seg
        self.eps = eps

        assert optim_type in OPTIM_TYPES
        self.optim_type = optim_type

        self.num_stages = num_stages
        self.add_noise = add_noise
        self.add_noise_init = add_noise  # backup values
        self.noise_thres = noise_thres
        self.color_thres = color_thres
        self.pts_dist_thres = pts_dist_thres

        # initailize parameters as zero or an empty set
        self.shapes = {}
        self.shape_groups = {}
        self.point_params = []
        self.color_params = []
        self.optimize_flag = []

        # initialize basic attributes related to dataset
        self.H: int = None
        self.W: int = None
        self.intrinsic: torch.Tensor = None

    @property
    def optimize_color(self):
        return self.optim_type != "Default"

    def set_intrinsic(self, hwf: HWF) -> None:
        # camera properties
        (
            self.H,
            self.W,
        ) = (
            hwf.height,
            hwf.width,
        )
        self.F = hwf.focal
        if self.proj_mode == "persp":  # perspective
            self.intrinsic = (
                torch.Tensor(
                    [
                        [hwf.focal, 0, hwf.width / 2],
                        [0, hwf.focal, hwf.height / 2],
                        [0, 0, 1],
                    ]
                )
                .float()
                .to(self.device)
            )

        else:  # orthographic
            self.intrinsic = (
                torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                .float()
                .to(self.device)
            )

    def init_properties_viewer(self, hwf: HWF) -> None:
        self.set_intrinsic(hwf)

    def init_properties(self, dataset: DataSet):
        """Reflect properties of given dataset"""

        # camera properties
        self.H, self.W, _ = dataset.get_HWF
        if self.proj_mode == "persp":  # perspective
            self.intrinsic = dataset.get_intrinsic.to(self.device)
        else:  # orthographic
            self.intrinsic = (
                torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                .float()
                .to(self.device)
            )

        # get extracted points or randomize points to set initial strokes
        if self.start_points is None:
            use_sfm = dataset.init_points and self.use_sfm
            if use_sfm:
                self.start_points, self.num_strokes = dataset.fps_from_sfm(
                    self.num_strokes
                )
            else:
                randomized = [
                    self.set_random_point(self.boundaries, device=self.device)
                    + self.center
                    for _ in range(self.num_strokes)
                ]
                self.start_points = torch.stack(randomized)

            # print final status of starting points
            print(
                f"Initialized curves: [num_strokes] {self.num_strokes} | [use_sfm] {use_sfm}"
            )

    def get_pts_3d(self, pt0: torch.Tensor, radius: float = 0.001) -> torch.Tensor:
        """Initialize the path starting with pt0"""

        if len(pt0.shape) == 2:
            start, end = pt0[0], pt0[1]
            # mid_points = rand_on_circle(start, end, num_points=2)
            mid_points = rand_on_line(start, end, num_points=2)
            pts_ls = [start] + mid_points + [end]
        else:
            radius = torch.ones([3]).to(self.device) * radius
            pts_ls = [pt0]

            for _ in range(self.num_segments):
                for _ in range(self.pts_per_seg - 1):
                    pt1 = pt0 + radius + torch.rand([3]).to(self.device) * self.gap
                    pts_ls.append(pt1)
                    pt0 = pt1

        # stack all points to control easily
        pts = torch.stack(pts_ls)  # [N_pts, 3]

        return pts

    def sort_strokes(self, pose: torch.Tensor):  # XXX fix consistency of colors
        """Sort strokes by the distance from the given camera location"""

        func = lambda x: euclidean_distance(torch.mean(x[0], dim=0), pose[:3, -1])
        params = list(zip(self.point_params, self.color_params))
        params.sort(key=func, reverse=True)
        for i, (pt, color) in enumerate(params):
            new_id = torch.Tensor([i]).int()
            self.shapes[pt].shape_ids = new_id
            self.shape_groups[pt].shape_ids = new_id
            self.shape_groups[pt].stroke_color = color

    def perspective(self, pts_3d: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Perspective projection."""

        extrinsic = torch.linalg.inv(pose)  # w2c
        if self.blender:
            extrinsic = blender2world(
                extrinsic[:3, :3], extrinsic[:3, -1:], self.device
            )

        pts_3d_hg = torch.concat([pts_3d, torch.ones_like(pts_3d[..., -1:])], dim=-1)
        world_matrix = self.intrinsic @ extrinsic[:3, ...]
        pts_2d_aug = pts_3d_hg @ world_matrix.T
        pts_2d_aug = pts_2d_aug / self.proj_scale
        pts_2d = pts_2d_aug[..., :-1] / pts_2d_aug[..., -1:]  # [N_pts, 2]

        return pts_2d.contiguous()

    def orthographic(self, pts_3d: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Orthographic projection."""

        extrinsic = torch.linalg.inv(pose)
        if self.blender:
            extrinsic = blender2world(
                extrinsic[:3, :3], extrinsic[:3, -1:], self.device
            )

        pts_3d_hg = torch.concat([pts_3d, torch.ones_like(pts_3d[..., -1:])], dim=-1)
        pts_3d_cam_T = extrinsic @ pts_3d_hg.T
        pts_2d = (self.intrinsic @ pts_3d_cam_T).T  # [N, 3]

        pts_2d = pts_2d[..., :-1]  # [N, 2]

        pts_2d[..., 0] = (pts_2d[..., 0] + 0.5) * self.W
        pts_2d[..., 1] = (pts_2d[..., 1] + 0.5) * self.H

        return pts_2d.contiguous()

    def initialize(self, pose: torch.Tensor) -> torch.Tensor:
        """Initialize strokes with the given view."""

        self.num_control_pts = torch.zeros(self.num_segments, dtype=torch.int32) + 2
        stroke_color = torch.Tensor([0.0, 0.0, 0.0, 1.0])

        if len(self.point_params) == 0:
            for pt0 in self.start_points:
                pt, path = self.init_path(pose, pt0)
                self.shapes[pt] = path
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.Tensor([len(self.shapes) - 1]),
                    fill_color=None,
                    stroke_color=stroke_color.clone(),
                )
                self.shape_groups[pt] = path_group
            self.optimize_flag = [True for _ in range(len(self.shapes))]
        else:
            self.load_paths(pose, stroke_color)

        img = self.render_warp(pose)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            self.H, self.W, 3, device=self.device
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)  # [H, W, C] -> [N, H, W, C]

        return img

    def init_path(
        self, pose: torch.Tensor, pt: torch.Tensor
    ) -> Tuple[torch.Tensor, pydiffvg.Path]:
        """Set a path based on the starting point"""

        # get 3D control points
        points = self.get_pts_3d(pt)

        path = pydiffvg.Path(
            num_control_points=self.num_control_pts,
            points=self.projection(points, pose),  # initialized
            stroke_width=torch.tensor(self.stroke_width, device=self.device),
            is_closed=False,
        )

        return points, path

    def load_paths(self, pose: torch.Tensor, stroke_color: torch.Tensor):
        """Load paths from given control points"""

        if len(self.color_params) == 0:
            color_params = [stroke_color for _ in range(len(self.point_params))]
        else:
            color_params = self.color_params

        new_idx = 0
        for pt, flag, color in zip(self.point_params, self.optimize_flag, color_params):
            if flag:
                path = pydiffvg.Path(
                    num_control_points=self.num_control_pts,
                    points=self.projection(pt, pose),  # initialized
                    stroke_width=torch.tensor(self.stroke_width, device=self.device),
                    is_closed=False,
                )
                self.shapes[pt] = path
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.Tensor([new_idx]).int(),
                    fill_color=None,
                    stroke_color=color,
                )
                self.shape_groups[pt] = path_group
                new_idx += 1

    def render_warp(self, pose: torch.Tensor) -> torch.Tensor:
        """Render sketches with projected 2D points."""

        if self.optim_type == "RGB":
            for group in self.shape_groups.values():
                group.stroke_color.data[:3].clamp_(0.0, 1.0)
                group.stroke_color.data[-1].clamp_(
                    1.0, 1.0
                )  # to force fully opaque strokes
        elif self.optim_type == "RGBA":
            for group in self.shape_groups.values():
                group.stroke_color.data.clamp_(0.0, 1.0)
        elif self.optim_type == "Alpha":
            for group in self.shape_groups.values():
                group.stroke_color.data[:3].clamp_(0.0, 0.0)  # to force black strokes
                group.stroke_color.data[-1].clamp_(0.0, 1.0)  # opacity

        # apply 2D projection with given position
        for pt, path in self.shapes.items():
            path.points = self.projection(pt, pose)

        # uncomment if you want to add random noise
        if self.add_noise:
            if random.random() > self.noise_thres:
                eps = self.eps * min(self.W, self.H)
                for path in self.shapes.values():
                    path.points.data.add_(eps * torch.randn_like(path.points))

        shapes_2d = list(self.shapes.values())
        shape_groups = list(self.shape_groups.values())
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.W, self.H, shapes_2d, shape_groups
        )

        _render = pydiffvg.RenderFunction.apply

        img = _render(
            self.W,  # width
            self.H,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,
            *scene_args,
        )

        return img

    def sketch(self, pose: torch.Tensor) -> torch.Tensor:
        # self.sort_strokes(pose)  # XXX remove this if needed
        sketch = self.render_warp(pose)

        alpha = sketch[..., -1:]
        sketch = sketch[..., :3] * alpha + (1.0 - alpha)
        sketch = sketch.unsqueeze(0)  # [N, H, W, 3]

        return sketch

    def save_svg(self, fname: str):
        shapes = list(self.shapes.values())
        shape_groups = list(self.shape_groups.values())
        pydiffvg.save_svg(fname, self.W, self.H, shapes, shape_groups)

    def cleaning(self, show: bool = True):
        counts = 0
        for i, pts in enumerate(self.point_params):
            mean_dist = get_mean_dist(pts)
            if mean_dist <= self.pts_dist_thres:
                self.inactive_stroke(i)
                counts += 1
        self.num_strokes -= counts
        if show:
            print(f"number of strokes: {len(self.point_params)} -> {self.num_strokes}")

    def inactive_stroke(self, idx: int):
        self.point_params[idx].requires_grad = False
        self.optimize_flag[idx] = False

    def set_point_params(self) -> List[torch.Tensor]:
        if len(self.point_params) == 0:
            self.point_params = []
            for i, pts in enumerate(self.shapes.keys()):
                if self.optimize_flag[i]:
                    pts.requires_grad = True
                    self.point_params.append(pts)

        return self.point_params

    def set_color_params(self) -> List[torch.Tensor]:
        if len(self.color_params) == 0:
            self.color_params = []
            for i, group in enumerate(self.shape_groups.values()):
                if self.optimize_flag[i]:
                    group.stroke_color.requires_grad = True
                    self.color_params.append(group.stroke_color)

        return self.color_params

    def set_random_noise(self, save: bool = False):
        self.add_noise = False if save else self.add_noise_init

    def load_state_dict(self, ckpt: Dict[str, Any]):
        self.point_params = ckpt["point_params"]
        self.color_params = ckpt["color_params"]
        self.optimize_flag = ckpt["optimize_flag"]

    def state_dict(self) -> Dict[str, List[Any]]:
        states = {
            "point_params": self.point_params,
            "color_params": self.color_params,
            "optimize_flag": self.optimize_flag,
        }

        return states

    def gui(self) -> None:
        pass


class CurveOptimizer:
    def __init__(
        self,
        module: CurveRenderer,
        point_lr: float = 1.0,
        color_lr: float = 0.01,
    ):
        """Optimizer used in CLIPasso.
        mainly referred to https://github.com/yael-vinker/CLIPasso/blob/main/models/painter_params.py
        """

        # renderer to optimize
        self.module = module
        # variables related to an optimizer
        self.point_lr = point_lr
        self.color_lr = color_lr
        self.optim_color = module.optimize_color

        # print status
        print(f"Optimize colors: {self.optim_color}")

    def initialize(self):
        self.point_optim = optim.Adam(self.module.set_point_params(), lr=self.point_lr)
        if self.optim_color:
            self.color_optim = optim.Adam(
                self.module.set_color_params(), lr=self.color_lr
            )

    def zero_grad(self):
        self.point_optim.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()

    def step(self):
        self.point_optim.step()
        if self.optim_color:
            self.color_optim.step()

    def state_dict(self) -> Dict[str, Any]:
        params = {}
        params["point"] = self.point_optim.state_dict()
        if self.optim_color:
            params["color"] = self.color_optim.state_dict()

        return params

    def load_state_dict(self, ckpt: Dict[str, Any]):
        self.point_optim.load_state_dict(ckpt["point"])
        if self.optim_color and "color" in ckpt.keys():
            self.color_optim.load_state_dict(ckpt["color"])

    def get_lr(self) -> float:
        return self.point_optim.param_groups[0]["lr"]
