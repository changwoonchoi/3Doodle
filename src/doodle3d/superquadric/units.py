import torch
import torch.nn as nn
from pytorch3d.transforms import quaternion_to_matrix as quat2rot

from typing import List, Dict, Union

from doodle3d.superquadric.contours import *
from doodle3d.utils.misc import RENDER_TYPES, get_rand_tensor, get_euler_angle
from doodle3d.utils.math_utils import safe_pow, euler2quat


class Superquadric(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        center: torch.Tensor = None,
        alpha: List[float] = [0.5, 1.5],
        epsilon: List[float] = [0.1, 1.5],
        gamma: float = 5.0,
        bias: float = 50.0,
        sigma_scale: float = 5.0,
        beta: float = 4.0,
        rendering_type: str = "contour_3doodle",
        safe: bool = True,
        **kwargs,
    ):
        """Superquadric module

        Args:
            alpha [float, float]:           scale factor range to initialize (min, max)
            epsilon [float, float]:         shape factor range (min, max)
        """

        super(Superquadric, self).__init__()

        assert len(center) == 3 and len(alpha) == 2 and len(epsilon) == 2
        assert rendering_type in RENDER_TYPES
        self.rendering_type = rendering_type

        self.device = device

        # range of epsilon values
        # self.e_min, self.e_max = 0.1, 1.5
        self.e_min, self.e_max = epsilon

        # parameters to optimize
        # all parameters except translation are randomly initialized
        self.translation = nn.Parameter(center)
        self.rotation = nn.Parameter(euler2quat(get_euler_angle(randomly=True)))
        self.alpha = nn.Parameter(
            get_rand_tensor([3], min=alpha[0], max=alpha[1], open_interval=False)
        )
        self.epsilon = nn.Parameter(
            get_rand_tensor([2], min=epsilon[0], max=epsilon[1], open_interval=True)
        )

        self.gamma = gamma
        self.beta = beta

        self.bias = bias
        self.sigma_scale = sigma_scale

        if self.rendering_type == "contour_3doodle_adaptive":
            self.gamma_min, self.gamma_max = kwargs["gamma_range"]
            self.bias_min, self.bias_max = kwargs["bias_range"]
            self.scale_min, self.scale_max = kwargs["scale_range"]

        self.pow = safe_pow if safe else torch.pow

    def clamp_epsilon(self):
        data = self.epsilon.data
        data = data.clamp(self.e_min, self.e_max)
        self.epsilon.data = data

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        rotmat = quat2rot(self.rotation)
        # x_tf = x @ rotmat.T - self.translation[None, ...]
        x_tf = (x - self.translation[None, ...]) @ rotmat.T
        return x_tf

    def superquadric_val(self, x: torch.Tensor) -> torch.Tensor:
        """Superquadric function."""

        x_tf = self.transform(x)

        eps0, eps1 = self.epsilon
        ax, ay, az = (
            x_tf[..., 0] / self.alpha[0],
            x_tf[..., 1] / self.alpha[1],
            x_tf[..., 2] / self.alpha[2],
        )

        left = self.pow(
            self.pow((ax**2), (1.0 / eps1)) + self.pow((ay**2), (1.0 / eps1)),
            (eps1 / eps0),
        )
        right = self.pow((az**2), (1.0 / eps0))
        f = left + right

        return f

    @torch.no_grad()
    def superquadric_normal(self, x: torch.Tensor) -> torch.Tensor:
        """Normal of superquadric."""

        x_tf = self.transform(x)

        a0, a1, a2 = self.alpha
        eps0, eps1 = self.epsilon
        ax, ay, az = x_tf[..., 0] / a0, x_tf[..., 1] / a1, x_tf[..., 2] / a2
        ax2, ay2, az2 = ax**2, ay**2, az**2

        # we can ignore 2 / eps0 term since we need to normalize the normal vector.
        common_term = self.pow(
            self.pow(ax2, (1.0 / eps1)) + self.pow(ay2, (1.0 / eps1)), (eps1 / eps0 - 1)
        )
        n0 = ax * self.pow(ax2, (1.0 / eps1 - 1)) * (1.0 / a0) * common_term
        n1 = ay * self.pow(ay2, (1.0 / eps1 - 1)) * (1.0 / a1) * common_term
        n2 = az * self.pow(az2, (1.0 / eps0 - 1)) * (1.0 / a2)

        n = torch.stack([n0, n1, n2], dim=-1)

        # multiply rotation term
        # d(f(Rx+t)) / dx = f'(Rx+t) * R
        rotmat = quat2rot(self.rotation)
        n = torch.matmul(n, rotmat)

        # normalize
        n = n / torch.norm(n, dim=-1, keepdim=True)

        return n

    def forward(self, x: torch.Tensor, viewdirs: torch.Tensor = None) -> torch.Tensor:
        self.clamp_epsilon()
        f = self.superquadric_val(x)
        if self.rendering_type == "sigma_ISCO":
            sigma = sigma_ISCO(f, gamma=self.gamma)
        elif self.rendering_type == "contour_ISCO":
            sigma = contour_ISCO(
                f, gamma=self.gamma, bias=self.bias, scale=self.sigma_scale
            )
        elif self.rendering_type == "contour_3doodle":
            n = self.superquadric_normal(x)
            sigma = contour_3doodle(
                f,
                n,
                viewdirs,
                gamma_1=self.gamma,
                bias=self.bias,
                scale=self.sigma_scale,
                beta=self.beta,
            )

        return sigma


class SuperquadricSet(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        centers: torch.Tensor = None,
        rendering_type: str = "contour_3doodle",
        adaptive_param_grad: bool = False,
        network_params: Dict[str, Union[List, float]] = None,
    ):
        """Set of superquadrics."""

        super(SuperquadricSet, self).__init__()

        self.device = device

        assert rendering_type in RENDER_TYPES
        self.rendering_type = rendering_type
        self.adaptive_param_grad = adaptive_param_grad

        network_params.update(
            {
                "device": self.device,
                "rendering_type": self.rendering_type,
            }
        )
        self.network_params = network_params
        self.superquadrics = [
            Superquadric(center=c, **network_params).to(self.device) for c in centers
        ]

        self.gamma = network_params["gamma"]
        self.bias = network_params["bias"]
        self.sigma_scale = network_params["sigma_scale"]
        self.beta = network_params["beta"]

        if self.rendering_type == "contour_3doodle_adaptive":
            self.adaptive_params = {
                "gamma_min": network_params["gamma_range"][0],
                "gamma_max": network_params["gamma_range"][1],
                "bias_min": network_params["bias_range"][0],
                "bias_max": network_params["bias_range"][1],
                "scale_min": network_params["scale_range"][0],
                "scale_max": network_params["scale_range"][1],
            }

        # add parameters to optimize
        for i, unit in enumerate(self.superquadrics):
            self.register_parameter(f"translation_{i}", unit.translation)
            self.register_parameter(f"rotation_{i}", unit.rotation)
            self.register_parameter(f"alpha_{i}", unit.alpha)
            self.register_parameter(f"epsilon_{i}", unit.epsilon)

    def add_single_superquadric(
        self,
        sq: Superquadric = None,
        scale_decay_factor: float = 1.0,
        debug: bool = False,
    ):
        """For coarse-to-fine iterative adding strategy"""

        if sq is None:
            a0, a1 = self.network_params["alpha"]
            kwargs = self.network_params.copy()
            kwargs.update({"alpha": [a0 * scale_decay_factor, a1 * scale_decay_factor]})
            sq = Superquadric(**kwargs).to(self.device)

        self.superquadrics.append(sq)
        idx = len(self.superquadrics) - 1

        self.register_parameter(f"translation_{idx}", sq.translation)
        self.register_parameter(f"rotation_{idx}", sq.rotation)
        self.register_parameter(f"alpha_{idx}", sq.alpha)
        self.register_parameter(f"epsilon_{idx}", sq.epsilon)

        if debug:
            CRED = "\033[101m"
            CEND = "\033[0m"
            print()
            print(CRED + f"Added {len(self.superquadrics)}th superquadric" + CEND)
            print()

    def forward(
        self, x: torch.Tensor, viewdirs: torch.Tensor = None, nan_to_num: bool = True
    ) -> torch.Tensor:
        """Merge superquadrics"""

        for superquadric in self.superquadrics:
            superquadric.clamp_epsilon()

        all_f = torch.stack(
            [sq.superquadric_val(x) for sq in self.superquadrics], dim=0
        )
        f, mask = torch.min(all_f, dim=0)
        if self.rendering_type in ["contour_3doodle", "contour_3doodle_adaptive"]:
            all_n = torch.stack(
                [sq.superquadric_normal(x) for sq in self.superquadrics], dim=0
            )
            normal = all_n.gather(
                dim=0, index=mask[None, ..., None].repeat(2, 1, 1, 3)
            )[0]
            if self.rendering_type == "contour_3doodle_adaptive":
                all_epsilon_mean = torch.stack(
                    [sq.epsilon.mean() for sq in self.superquadrics], dim=0
                )
                all_alpha_mean = torch.stack(
                    [sq.alpha.mean() for sq in self.superquadrics], dim=0
                )
                epsilon_mean = all_epsilon_mean[mask]
                alpha_mean = all_alpha_mean[mask]

        if self.rendering_type == "sigma_ISCO":
            sigma = sigma_ISCO(f, self.gamma)
        elif self.rendering_type == "contour_ISCO":
            sigma = contour_ISCO(f, self.gamma, self.bias, self.sigma_scale)
        elif self.rendering_type == "contour_3doodle":
            sigma = contour_3doodle(
                f, normal, viewdirs, self.gamma, self.bias, self.sigma_scale, self.beta
            )
        elif self.rendering_type == "contour_3doodle_adaptive":
            sigma = contour_3doodle_adaptive(
                alpha_mean,
                epsilon_mean,
                f,
                normal,
                viewdirs,
                self.beta,
                self.adaptive_param_grad,
                **self.adaptive_params,
            )
        else:
            raise ValueError("Invalid rendering type.")

        if nan_to_num:
            sigma = torch.nan_to_num(sigma)

        return sigma

    def gui(self) -> None:

        import polyscope.imgui as psim

        if psim.TreeNode("Superquadrics"):

            _, self.gamma = psim.InputFloat("gamma##sq", self.gamma)
            _, self.beta = psim.InputFloat("beta##sq", self.beta)

            if self.rendering_type == "contour_3doodle_adaptive":
                _, self.adaptive_params["gamma_min"] = psim.InputFloat(
                    "gamma_min##sq", self.adaptive_params["gamma_min"]
                )
                _, self.adaptive_params["gamma_max"] = psim.InputFloat(
                    "gamma_max##sq", self.adaptive_params["gamma_max"]
                )
                _, self.adaptive_params["bias_min"] = psim.InputFloat(
                    "bias_min##sq", self.adaptive_params["bias_min"]
                )
                _, self.adaptive_params["bias_max"] = psim.InputFloat(
                    "bias_max##sq", self.adaptive_params["bias_max"]
                )
                _, self.adaptive_params["scale_min"] = psim.InputFloat(
                    "scale_min##sq", self.adaptive_params["scale_min"]
                )
                _, self.adaptive_params["scale_max"] = psim.InputFloat(
                    "scale_max##sq", self.adaptive_params["scale_max"]
                )

            psim.TreePop()
