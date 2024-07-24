import time
import signal
import math
from argparse import Namespace
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import pydiffvg

import polyscope as ps
import polyscope.imgui as psim

from doodle3d.sketcher import Doodle
from doodle3d.utils.misc import signal_handler, fov2focal, HWF, pose_to_rays
from doodle3d.utils.arguments import parse_args
from doodle3d.utils.io import load_config_testing

from nerfacc.estimators.occ_grid import OccGridEstimator

MAX_DEPTH = 10.0


class Viewer:
    def __init__(self, args: Namespace) -> None:

        pydiffvg.set_use_gpu(torch.cuda.is_available())
        self.device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

        self.config, exp_dir = load_config_testing(self.device, args, show=True)

        # print path of directory to save
        print(f"[exp_dir] ./logs/{exp_dir}")

        self.sketcher = Doodle(**self.config["method"])
        self.renderer = None
        self.occ_grid = None
        self.sq_enabled = True
        self.render_step_size = 1e-2
        self.alpha_thre = 0.01

        ps.set_program_name("3Doodle")
        ps.init()
        self.ps_init()

        ps.set_user_callback(self.ps_callback)
        ps.show()

    def ps_init(self) -> None:
        """
        Initialize Polyscope
        """
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("z_up")
        ps.set_max_fps(120)
        # Anti-aliasing
        ps.set_SSAA_factor(4)
        # Prevent polyscope from changing scales (including Gizmo!)
        ps.set_automatically_compute_scene_extents(False)
        ps.look_at(4.0 * np.ones(3), np.array([0.0, 0.0, 0.0]))

        self.update_render_sizes()
        self.init_render_buffer()

        self.last_time = time.time()

    def init_renderer(self, hwf: HWF):
        self.sketcher.prepare_viewer(hwf, only_sq=False)
        self.renderer = self.sketcher.renderer
        # In any case, contours are not rendered in a vanilla way so we disable
        # them here!
        self.renderer.use_contour = False
        self.renderer.eval()

        if self.renderer.sq_renderer is not None:
            # scene
            aabb = torch.tensor(
                [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5],
                device=self.device,
            )
            grid_resolution = 128
            # render parameters
            self.cone_angle = 0.0
            self.occ_grid = OccGridEstimator(
                roi_aabb=aabb, resolution=grid_resolution, levels=1
            ).to(self.device)

            self.update_occupancy()

            # Uncomment to display the occupancy point cloud
            # binaries = self.occ_grid.binaries
            # grid_coords = self.occ_grid.grid_coords
            # occupied = grid_coords[binaries.flatten()]
            # xyzs_w = self.occ_grid.aabbs[0, :3] + occupied / grid_resolution * (
            #     self.occ_grid.aabbs[0, 3:] - self.occ_grid.aabbs[0, :3]
            # )
            # ps.register_point_cloud("occupied", xyzs_w.cpu().numpy())

    def update_occupancy(self, n_iter: int = 1000):
        def occ_eval_fn(x):
            view_dir = 2.0 * torch.rand_like(x) - 1.0
            view_dir /= torch.linalg.norm(view_dir, dim=-1, keepdim=True)
            density = F.relu(
                self.sketcher.renderer.sq_renderer.network(x.unsqueeze(1), view_dir)
            ).squeeze(1)
            return density * self.render_step_size

        # Update occupancy grid: that's quite DIY :D
        for i in tqdm(range(n_iter)):
            self.occ_grid.update_every_n_steps(
                step=i, occ_eval_fn=occ_eval_fn, occ_thre=1e-3, ema_decay=0.999
            )

    def update_render_sizes(self) -> None:
        self.window_size = ps.get_window_size()
        self.buffer_size = (
            int(self.window_size[0]),
            int(self.window_size[1]),
        )

        # Update the renderer's intrinsics
        ps_view_camera_parameters = ps.get_view_camera_parameters()
        WIDTH = self.window_size[0]
        HEIGHT = self.window_size[1]
        focal = fov2focal(
            ps_view_camera_parameters.get_fov_vertical_deg() * math.pi / 180.0,
            HEIGHT,
        )
        hwf = HWF(height=HEIGHT, width=WIDTH, focal=focal)

        if self.renderer is None:
            self.init_renderer(hwf)
        else:
            self.renderer.set_intrinsic(hwf)

    def init_render_buffer(self) -> None:
        self.render_buffer_quantity = ps.add_raw_color_alpha_render_image_quantity(
            "render_buffer",
            MAX_DEPTH
            * np.ones((self.buffer_size[1], self.buffer_size[0]), dtype=float),
            np.zeros((self.buffer_size[1], self.buffer_size[0], 4), dtype=float),
            enabled=True,
            allow_fullscreen_compositing=True,
        )

        self.render_buffer = ps.get_quantity_buffer("render_buffer", "colors")

    def ps_callback(self) -> None:

        new_time = time.time()
        self.fps = 1.0 / (new_time - self.last_time)
        self.last_time = new_time

        self.gui()
        if self.renderer is not None:
            self.renderer.gui()

        self.draw()

    @torch.no_grad()
    def gui(self) -> None:
        psim.Text(f"fps: {self.fps:.4f};")

        if self.occ_grid is not None:
            if psim.Button("Update occupancy"):
                self.update_occupancy()

        if self.renderer is not None and self.renderer.sq_renderer is not None:
            _, self.sq_enabled = psim.Checkbox(
                "Render superquadrics##viewer", self.sq_enabled
            )
        if self.renderer.sq_renderer is not None and psim.TreeNode(
            "Rendering Options##viewer"
        ):
            _, self.render_step_size = psim.SliderFloat(
                "Step size##viewer", self.render_step_size, v_min=0.001, v_max=0.1
            )
            _, self.alpha_thre = psim.SliderFloat(
                "Alpha threshold##viewer",
                self.alpha_thre,
                v_min=0.001,
                v_max=0.1,
            )
            psim.TreePop()

    @torch.no_grad()
    def draw(self) -> None:

        # Handle window resize
        if ps.get_window_size() != self.window_size:
            self.update_render_sizes()
            self.init_render_buffer()

        # --------------------------
        # PROCESS CAMERA
        # --------------------------

        ps_view_camera_parameters = ps.get_view_camera_parameters()

        c2w = torch.linalg.inv(torch.tensor(ps_view_camera_parameters.get_view_mat()))

        window_size = ps.get_window_size()
        WIDTH = window_size[0]
        HEIGHT = window_size[1]
        focal = fov2focal(
            ps_view_camera_parameters.get_fov_vertical_deg() * math.pi / 180.0,
            HEIGHT,
        )

        rays = pose_to_rays(
            pose=c2w, width=WIDTH, height=HEIGHT, focal=focal, device=self.device
        )

        # --------------------------
        # RENDER
        # --------------------------

        # a. Bezier curves
        img_sketch, _ = self.renderer(pose=c2w, rays=rays, only_sq=False)
        img = img_sketch.squeeze(0)

        # b. Superquadrics (with occupancy grid)
        if self.renderer.sq_renderer is not None and self.sq_enabled:
            img_contour = self.renderer.sq_renderer.render_with_occupancy(
                rays=rays,
                occ_grid=self.occ_grid,
                cone_angle=self.cone_angle,
                alpha_thre=self.alpha_thre,
                render_step_size=self.render_step_size,
                near=0.0,
                far=1.0e10,
            ).reshape(*img.shape)

            # Union
            img = 1 - (1 - img + 1 - img_contour).clamp(0.0, 1.0)

        # Update render buffer
        rendered_image = torch.cat(
            [
                img,
                torch.ones((img.shape[0], img.shape[1], 1), device=img.device),
            ],
            dim=-1,
        )

        self.render_buffer.update_data_from_device(rendered_image)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    Viewer(args)
