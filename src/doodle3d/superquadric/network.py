import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Tuple, Union

from doodle3d.superquadric.units import SuperquadricSet
from doodle3d.dataset.base import DataSet
from doodle3d.utils.misc import RENDER_TYPES, HWF

from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.volrend import rendering


class SQRenderer(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        chunk: int = 1024 * 32,
        rendering_type: str = "contour_3doodle",
        network_params: Dict[str, Union[List, float]] = dict(),
        use_sfm: bool = True,
        n_units: int = 1,
        n_samples: int = 192,
        color: List[float] = [0.0, 0.0, 0.0],
        white_bkgd: bool = True,
        adaptive_param_grad: bool = False,
    ):
        super().__init__()

        self.device = device
        self.chunk = chunk
        self.n_samples = n_samples
        self.color = torch.Tensor(color).to(self.device)
        self.white_bkgd = white_bkgd

        self.k_extract = ["rgb_map", "disp_map", "acc_map"]

        assert rendering_type in RENDER_TYPES
        self.use_sfm = use_sfm
        self.n_units = n_units
        self.rendering_type = rendering_type
        self.adaptive_param_grad = adaptive_param_grad
        self.network_params = network_params

        self.network: SuperquadricSet = None  # not initialized yet

    def set_intrinsic(self, hwf: HWF) -> None:
        # camera properties
        (
            self.H,
            self.W,
        ) = (
            hwf.height,
            hwf.width,
        )

    def init_properties_viewer(self, hwf: HWF) -> None:
        self.set_intrinsic(hwf)

        if self.network is None:
            self.network = SuperquadricSet(
                self.device,
                torch.zeros((self.n_units, 3)),
                self.rendering_type,
                self.adaptive_param_grad,
                self.network_params,
            ).to(self.device)

    def init_properties(self, dataset: DataSet):
        """Reflect properties of given dataset."""

        self.H, self.W, _ = dataset.get_HWF
        self.near, self.far = dataset.near, dataset.far

        # set superquadrics with given information of dataset
        if self.network is None:
            centers, _ = dataset.farthest_sampling(self.n_units)
            use_sfm = (centers is not None) and self.use_sfm
            if len(centers.shape) == 3:
                centers = centers.mean(dim=1)  # [N, 2, 3] -> [N, 3]
            if not use_sfm:
                centers = torch.rand(self.n_units, 3)
            self.network = SuperquadricSet(
                self.device,
                centers,
                self.rendering_type,
                self.adaptive_param_grad,
                self.network_params,
            ).to(self.device)

            # print final status of initial superquadrics
            print(
                f"Initialized contours: [n_units] {self.n_units} | [use_sfm] {use_sfm}"
            )

    def add_superquadric(self, scale_decay_factor: float = 1.0, debug: bool = False):
        self.network.add_single_superquadric(
            scale_decay_factor=scale_decay_factor, debug=debug
        )

    def render_with_occupancy(
        self,
        rays: torch.Tensor,
        occ_grid: OccGridEstimator,
        cone_angle: float,
        alpha_thre: float,
        render_step_size: float,
        near: float,
        far: float,
    ):
        """
        Render the superquadrics using an occupancy grid
        """
        # separate orientation and direction of rays
        rays = rays.reshape(-1, 6)

        rays_o = rays[..., :3]
        rays_d = rays[..., 3:]

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigma = F.relu(self.network(positions.unsqueeze(1), t_dirs)).squeeze(1)
            return sigma

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            if t_starts.shape[0] == 0:
                rgbs = torch.empty((0, 3), device=t_starts.device)
                sigmas = torch.empty((0, 1), device=t_starts.device)
            else:
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                raw = self.network(positions.unsqueeze(1), t_dirs).squeeze(1)
                sigmas = F.relu(raw)
                rgbs = torch.zeros(raw.shape[0], 3).to(self.device)
                rgbs[..., :] = self.color
            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = occ_grid.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=near,
            far_plane=far,
            render_step_size=render_step_size,
            stratified=False,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=rays_o.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=torch.tensor([1.0, 1.0, 1.0]).to(self.device),
        )

        return rgb

    def render(
        self,
        rays: torch.Tensor,
        use_viewdirs: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor]]:
        # separate orientation and direction of rays
        rays = rays.reshape(-1, 6)

        # provide ray directions as input
        rays_d = rays[..., 3:]
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

        near, far = self.near * torch.ones_like(
            rays_d[..., :1]
        ), self.far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays, near, far], dim=-1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], dim=-1)

        # Render and reshape
        all_ret = self.batchify_rays(rays)
        for k, v in all_ret.items():
            all_ret[k] = v.reshape(self.H, self.W, -1)

        extracts = {k: v for k, v in all_ret.items() if k in self.k_extract}
        others = {k: v for k, v in all_ret.items() if k not in self.k_extract}

        return extracts, others

    def batchify_rays(self, rays_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Render rays in smaller minibatches to avoid OOM."""

        n_rays, _ = rays_flat.shape
        all_ret: Dict[str, List] = {}
        for i in range(0, n_rays, self.chunk):
            ret = self.render_rays(rays_flat[i : i + self.chunk])
            for k, v in ret.items():
                if k in all_ret:
                    all_ret[k] = torch.cat([all_ret[k], v], dim=0)
                else:
                    all_ret[k] = v

        return all_ret

    def render_rays(self, rays_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute superquadrics using given modules."""

        n_rays, _ = rays_batch.shape
        rays_o, rays_d = rays_batch[..., 0:3], rays_batch[..., 3:6]  # [N_rays, 3] each
        viewdirs = rays_batch[..., -3:] if rays_batch.shape[-1] > 8 else None

        bounds = torch.reshape(rays_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1, 1]

        t_vals = torch.linspace(0.0, 1.0, steps=self.n_samples).to(self.device)
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([n_rays, self.n_samples])

        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]
        raw = self.network(pts, viewdirs)
        ret = self.raw2outputs_superquadric(raw, z_vals, rays_d)

        # for k in ret:
        #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
        #         print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def raw2outputs_superquadric(
        self,
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        eps: float = 1.0e-10,
        inf: float = 1.0e10,
    ) -> Dict[str, torch.Tensor]:
        """Convert raw to outputs"""

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(
            -act_fn(raw) * dists
        )

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.Tensor([inf]).expand(dists[..., :1].shape).to(self.device)],
            -1,
        )  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.zeros(raw.shape[0], raw.shape[1], 3).to(
            self.device
        )  # [N_rays, N_samples, 3]
        rgb[..., :] = self.color

        alpha = raw2alpha(raw, dists)  # [N_rays, N_samples]
        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [
                        torch.ones((alpha.shape[0], 1)).to(self.device),
                        1.0 - alpha + eps,
                    ],
                    -1,
                ),
                -1,
            )[..., :-1]
        )
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1.0 / torch.max(
            eps * torch.ones_like(depth_map).to(self.device),
            depth_map / torch.sum(weights, -1),
        )
        acc_map = torch.sum(weights, -1)

        if self.white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_dict = {
            "rgb_map": rgb_map,
            "disp_map": disp_map,
            "acc_map": acc_map,
            "weights": weights,
            "depth_map": depth_map,
        }

        return ret_dict

    def gui(self) -> None:

        self.network.gui()
