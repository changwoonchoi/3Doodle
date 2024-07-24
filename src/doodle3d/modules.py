import torch
import torch.nn as nn
import torch.optim as optim

from typing import Dict, Any, Tuple
from dataclasses import dataclass

from doodle3d.curve.network import CurveRenderer, CurveOptimizer
from doodle3d.superquadric.network import SQRenderer
from doodle3d.dataset.base import DataSet
from doodle3d.utils.misc import conditional_decorator, HWF


class Renderer(nn.Module):  # XXX optimize this later
    def __init__(
        self,
        device: str = "cuda:0",
        sq_pre_iters: int = 0,
        sq_freeze: bool = False,
        use_contour: bool = False,
        use_viewdirs: bool = False,
        curve_params: Dict[str, Any] = None,
        sq_params: Dict[str, Any] = None,
    ):
        """Main module to represent contours and sketches based on bezier curves and superquadrics"""

        super().__init__()

        self.device = device
        self.use_contour = use_contour
        self.use_viewdirs = use_viewdirs
        self.sq_pre_iters = sq_pre_iters
        self.sq_freeze = sq_freeze

        self.curve_renderer = CurveRenderer(device, **curve_params).to(self.device)
        if self.use_contour:
            # sq_params.update({"sq_adding_list": self.sq_adding_list})
            self.sq_renderer = SQRenderer(device, **sq_params).to(self.device)
        else:
            self.sq_renderer = None

        self.use_curve: bool = False

    def set_usage(self, only_sq: bool = False, pred: bool = False):
        self.use_curve = (not only_sq) or pred

    def get_optimizer(self, **kwargs):
        optimizer = Optimizer(
            self.curve_renderer, self.sq_renderer, self.sq_freeze, **kwargs
        )
        return optimizer

    def set_intrinsic(self, hwf: HWF) -> None:
        self.curve_renderer.set_intrinsic(hwf)
        if self.use_contour:
            self.sq_renderer.set_intrinsic(hwf)

    def init_properties_viewer(self, hwf: HWF):
        self.curve_renderer.init_properties_viewer(hwf)
        if self.use_contour:
            self.sq_renderer.init_properties_viewer(hwf)

    def init_properties(self, dataset: DataSet):
        self.curve_renderer.init_properties(dataset)
        if self.use_contour:
            self.sq_renderer.init_properties(dataset)

    def set_random_noise(self, save: bool = True):
        self.curve_renderer.set_random_noise(save=save)

    def initialize(self, pose: torch.Tensor) -> torch.Tensor:
        init = self.curve_renderer.initialize(pose)
        return init

    def save_svg(self, fname: str):
        self.curve_renderer.save_svg(fname)

    def clean_strokes(self):
        self.curve_renderer.cleaning()

    def forward(
        self,
        pose: torch.Tensor,
        rays: torch.Tensor,
        only_sq: bool = False,
    ) -> torch.Tensor:
        """Draw view-dependent and view-independent lines"""

        @conditional_decorator(torch.no_grad(), (not only_sq) and self.sq_freeze)
        def render_contours(
            rays: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            ret_dict, _ = self.sq_renderer.render(rays, self.use_viewdirs)
            contour = ret_dict["rgb_map"].unsqueeze(0)  # [H, W, C] -> [N, H, W, C]
            return contour, ret_dict

        # if self.use_contour:
        #     # scale_decay_factor = 0.9 ** (self.sq_adding_list.index(iters))
        #     scale_decay_factor = 1.0
        #     self.sq_renderer.add_superquadric(scale_decay_factor=scale_decay_factor)

        if only_sq:
            assert self.use_contour, "unable to compute contours"
            contour, ret_dict = render_contours(rays)
            del ret_dict["rgb_map"]

            return contour, ret_dict

        sketch = self.curve_renderer.sketch(pose.squeeze())
        ret_dict = {}
        if self.use_contour:
            ret_dict.update({"curves": sketch.squeeze()})  # [H, W, 3]
            contour, extracts = render_contours(rays)
            sketch = 1 - (1 - sketch + 1 - contour).clamp(0.0, 1.0)  # union
            ret_dict.update(extracts)

        return sketch, ret_dict

    def gui(self) -> None:
        self.curve_renderer.gui()
        if self.sq_renderer is not None:
            self.sq_renderer.gui()

    def state_dict(self) -> Dict[str, Any]:
        states = {}
        if self.use_curve:
            states.update({"curve": self.curve_renderer.state_dict()})
        if self.use_contour:
            states.update({"superquadric": self.sq_renderer.state_dict()})

        return states

    def load_state_dict(self, ckpt: Dict[str, Any]):
        if self.use_curve and "curve" in ckpt.keys():
            self.curve_renderer.load_state_dict(ckpt["curve"])
        if self.use_contour or "superquadric" in ckpt.keys():
            self.sq_renderer.load_state_dict(ckpt["superquadric"])


class Optimizer:
    def __init__(
        self,
        curve: CurveRenderer,
        sq_renderer: SQRenderer = None,
        sq_freeze: bool = False,
        point_lr: float = 1.0,
        color_lr: float = 0.1,
        render_lr: float = 0.01,
    ):
        """Main class of the optimizer"""

        self.sq_freeze = sq_freeze
        self.use_contour = sq_renderer is not None

        self.grad_curve: bool = False
        self.grad_contour: bool = False

        self.curve_optim = CurveOptimizer(
            module=curve,
            point_lr=point_lr,
            color_lr=color_lr,
        )
        if self.use_contour:
            self.sq_optim = optim.Adam(sq_renderer.parameters(), lr=render_lr)

    def get_scheduler(self, steps: int, min_lr: float):
        assert self.use_contour, "scheduler is not required if only rendering curves."
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.sq_optim, T_max=steps, eta_min=min_lr
        )

        return scheduler

    def set_grads(self, only_sq: bool = False):
        self.grad_contour = only_sq or (self.use_contour and (not self.sq_freeze))
        self.grad_curve = not only_sq

    def initialize(self):
        self.curve_optim.initialize()

    def zero_grad(self):
        if self.grad_curve:
            self.curve_optim.zero_grad()
        if self.grad_contour:
            self.sq_optim.zero_grad()

    def step(self):
        if self.grad_curve:
            self.curve_optim.step()
        if self.grad_contour:
            self.sq_optim.step()

    def state_dict(self) -> Dict[str, Any]:
        states = {}
        if self.grad_curve:
            states.update({"curve_optim": self.curve_optim.state_dict()})
        if self.use_contour:
            states.update({"sq_optim": self.sq_optim.state_dict()})

        return states

    def load_state_dict(self, ckpt: Dict[str, Any]):
        if self.grad_curve and "curve_optim" in ckpt.keys():
            self.curve_optim.load_state_dict(ckpt["curve_optim"])
        if self.grad_contour and "sq_optim" in ckpt.keys():
            self.sq_optim.load_state_dict(ckpt["sq_optim"])
