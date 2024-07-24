import torch

import os
import cv2
import imageio
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from doodle3d.loss.main import Loss
from doodle3d.modules import Renderer, Optimizer
from doodle3d.dataset.base import DataSet
from doodle3d.utils.misc import HWF


class Doodle:
    def __init__(
        self,
        # basic parameters
        device: str = "cuda:0",
        exp_dir: str = None,
        ckpt_dir: str = None,
        plot_freq: int = 50,
        save_freq: int = 50,
        desc_freq: int = 5,
        eval_gap: int = 20,
        # parameters to visualize status
        save_init: bool = True,
        save_trains: bool = True,
        # parameters of dataloader
        batch_size: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
        worker_init: bool = True,
        # parameters to train
        max_iters: int = 20000,
        num_stage: int = 1,
        lr: float = 1.0,
        color_lr: float = 0.01,
        render_lr: float = 1.0e-2,
        use_contour: bool = False,
        only_contour: bool = True,
        use_viewdirs: bool = False,
        # parameters related to sketches
        clean: bool = False,
        sq_pre_iters: int = 0,
        sq_freeze: bool = False,
        curve_params: Dict[str, Any] = None,
        sq_params: Dict[str, Any] = None,
        # parameters related to CLIP and loss function
        loss_params: Dict[str, Any] = None,
        # parameters for evaluation
        save_svg: bool = True,
        **kwargs,
    ):
        """Main class to get sketch in a given 3D scene"""

        self.device = device
        self.plot_freq = plot_freq
        self.save_freq = save_freq
        self.desc_freq = desc_freq
        self.eval_gap = eval_gap

        self.save_init = save_init
        self.save_trains = save_trains

        self.cur_iter = 0  # initialized
        self.max_iters = max_iters
        self.num_stage = num_stage

        self.dl_args = {  # XXX if using dataloader of the trainset
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": shuffle,
            "worker_init": worker_init,
        }
        self.batch_size = batch_size

        self.only_contour = only_contour
        self.use_contour = True if self.only_contour else use_contour
        self.sq_pre_iters = sq_pre_iters

        self.clean = clean

        renderer_kwargs = {
            "device": self.device,
            "use_contour": self.use_contour,
            "sq_pre_iters": self.sq_pre_iters,
            "sq_freeze": sq_freeze,
            "use_viewdirs": use_viewdirs,
            "curve_params": curve_params,
            "sq_params": sq_params,
        }
        self.renderer = Renderer(**renderer_kwargs).to(self.device)

        self.optim_kwargs = {
            "point_lr": lr,
            "color_lr": color_lr,
            "render_lr": render_lr,
        }
        self.optimizer: Optimizer = None

        self.loss = Loss(self.device, **loss_params)
        self.loss_last = None
        self.loss_min = None

        self.save_svg = save_svg

        self.root = f"./logs/{exp_dir}"
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.ckpt_dir = ckpt_dir
        self.loaded = False

    def update_best(self) -> bool:
        if self.only_contour or (not self.use_contour):
            return True
        return self.cur_iter > self.sq_pre_iters

    def load_ckpts(self, ckpt_dir: str, pose: torch.Tensor, train: bool = True):
        ckpt = torch.load(ckpt_dir, map_location=self.device)
        self.cur_iter = ckpt["cur_iter"] + 1
        self.loss_last = ckpt["loss"]

        self.renderer.load_state_dict(ckpt["drawer"])
        if self.clean and (not train):
            self.renderer.clean_strokes()

        init = self.renderer.initialize(pose)
        if train:
            self.optimizer.initialize()
            self.optimizer.load_state_dict(ckpt["optimizer"])
        # print messages
        print("Loaded checkpoints from the given path.")
        self.loaded = True

        return init

    def save_ckpts(self, ckpt_dir: str, best: bool = False):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path = (
            f"{ckpt_dir}/best.ckpt" if best else f"{ckpt_dir}/{self.cur_iter:06d}.ckpt"
        )
        ckpt = {
            "drawer": self.renderer.state_dict(),
            "cur_iter": self.cur_iter,
            "loss": self.loss_last,
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(ckpt, ckpt_path)

    def initialize(self, init_pose: torch.Tensor) -> torch.Tensor:
        init = self.renderer.initialize(init_pose)
        self.optimizer.initialize()
        self.loaded = True

        return init

    def prepare_viewer(self, hwf: HWF, only_sq: bool = False) -> None:
        self.renderer.init_properties_viewer(hwf)
        self.renderer.set_random_noise(save=True)
        self.renderer.set_usage(only_sq, pred=True)

        init_pose = torch.eye(4).to(self.device)
        if self.ckpt_dir is not None:
            self.load_ckpts(self.ckpt_dir, init_pose, train=False)
        else:
            raise ValueError("Cannot prepare for viewer without a checkpoint!")

    def prepare(
        self, dataset: DataSet, only_sq: bool = False, train: bool = True
    ) -> torch.Tensor:
        """Prepare to optimize with the given dataset"""

        self.renderer.init_properties(dataset)
        if self.optimizer is None:
            self.optimizer = self.renderer.get_optimizer(**self.optim_kwargs)

        self.renderer.set_random_noise(save=True)

        self.renderer.set_usage(only_sq, pred=(not train))
        if train:
            self.optimizer.set_grads(only_sq)

        if not self.loaded:
            init_pose = dataset.poses[0].to(self.device)
            if self.ckpt_dir is not None:
                init = self.load_ckpts(self.ckpt_dir, init_pose, train=train)
            else:
                init = self.initialize(init_pose)
        else:
            init = None

        return init

    def learn(self, dataset: DataSet, test_dataset: DataSet, only_sq: bool = False):
        """Learn modules
        Cases:
            1. use only contours: optimize only superquadrics
            2. use only curves: optimize only bezier curves
            3. use both: optimize superquadrics -> add bezier curves
        """

        log_dir = f"{self.root}/train"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        init = self.prepare(dataset, only_sq, train=True)

        if self.save_init and (init is not None):
            init = init.detach()
            init = init.clamp(0.0, 1.0).squeeze().cpu().numpy() * 255.0
            imageio.imwrite(f"{log_dir}/init.png", init.astype(np.uint8))

        if only_sq:
            max_iters = self.max_iters if self.only_contour else self.sq_pre_iters
            self.train(dataset, test_dataset, max_iters, log_dir, only_sq=True)
        else:
            self.train(dataset, test_dataset, self.max_iters, log_dir, only_sq=False)

    def train(
        self,
        dataset: DataSet,
        test_dataset: DataSet,
        max_iters: int,
        log_dir: str,
        only_sq: bool = False,
    ):
        """Train modules
        pseudo code:
            initialize points in 3D

            for cur_iter in range(max_iters):
                img <- random images in training viewpoints
                compute S_3d
                S_2d <- projection(S_3d)  # orthogonal projection
                compute_loss(S_2d, img)
                loss.backward()
        """

        pbar = tqdm(range(self.cur_iter, max_iters), colour="green")
        for self.cur_iter in pbar:
            # load data
            item = dataset.next(batch_size=self.batch_size)
            rgbs, poses, rays = item["rgbs"], item["poses"], item["rays"]

            self.renderer.set_random_noise(
                self.cur_iter % self.save_freq == self.save_freq - 1
            )

            self.optimizer.zero_grad()
            sketches, _ = self.renderer(poses, rays, only_sq=only_sq)
            loss = self.loss(sketches, rgbs, train=True, only_sq=only_sq)
            loss.backward()
            self.optimizer.step()

            # update description
            if self.cur_iter % self.desc_freq == 0:
                eval_loss = 0.0 if self.loss_last is None else self.loss_last.item()
                desc = f"{self.cur_iter}/{self.max_iters-1} loss: {loss.item():.5f} | eval: {eval_loss:.5f}"
                pbar.set_description(desc)

            # evalutate to plot the results, plot initial images
            if (
                self.cur_iter % self.plot_freq == self.plot_freq - 1
                or self.cur_iter == 0
            ):
                self.eval(test_dataset, gap=self.eval_gap, only_sq=only_sq)

            # save current status
            if self.cur_iter % self.save_freq == self.save_freq - 1:
                self.save_ckpts(f"{log_dir}/checkpoints", best=False)
                if self.save_trains:
                    sketches = sketches.detach()
                    sketches = sketches.clamp(0.0, 1.0).squeeze().cpu().numpy() * 255.0
                    imageio.imwrite(
                        f"{log_dir}/intermediate.png", sketches.astype(np.uint8)
                    )

    @torch.no_grad()
    def eval(self, dataset: DataSet, gap: int = 10, only_sq: bool = False):
        """Evaluation"""

        log_dir = f"{self.root}/eval/{self.cur_iter:06d}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logs = {"inputs": [], "sketches": []}
        if only_sq:
            logs.update({"disp_map": [], "acc_map": []})
        elif self.use_contour:
            logs.update({"rgb_map": [], "disp_map": [], "acc_map": [], "curves": []})

        items = dataset.all_items
        all_rgbs, all_poses, all_rays = items["rgbs"], items["poses"], items["rays"]

        self.renderer.eval()

        losses = []

        pbar = tqdm(range(len(dataset)), colour="blue")
        pbar.set_description("Eval")
        for idx in pbar:
            rgbs, poses, rays = (
                all_rgbs[idx : idx + 1],
                all_poses[idx],
                all_rays[idx : idx + 1],
            )
            sketches, ret_dict = self.renderer(poses, rays, only_sq=only_sq)
            loss = self.loss(sketches, rgbs, train=False, only_sq=only_sq)
            losses.append(loss)

            if idx % gap == gap - 1:
                rgbs = rgbs.squeeze().cpu().numpy() * 255.0
                sketches = sketches.clamp(0.0, 1.0).squeeze().cpu().numpy() * 255.0
                logs["inputs"].append([idx, rgbs])
                logs["sketches"].append([idx, sketches])

                if self.use_contour:
                    for k, v in ret_dict.items():
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            # print(f"! [Numerical Error] {k} contains nan or inf.")
                            v = v.nan_to_num()
                        v = v.repeat(1, 1, 3) if v.shape[-1] == 1 else v  # [H, W, 3]
                        v = v.clamp(0.0, 1.0).squeeze().cpu().numpy() * 255.0
                        logs[k].append([idx, v])

        # update the latest evaluation loss
        self.loss_last = torch.mean(torch.Tensor(losses))

        # update minimum loss value
        if self.update_best():
            current_best = (self.loss_min is None) or (self.loss_last < self.loss_min)
            if current_best:
                self.loss_min = self.loss_last
                self.save_ckpts(self.root, best=True)
        else:
            current_best = False

        # save results
        for key, ls in logs.items():
            values = np.concatenate([v for _, v in ls], axis=1).astype(np.uint8)
            imageio.imwrite(f"{log_dir}/{key}.png", values)
            # update best results
            if key == "sketches" and current_best:
                imageio.imwrite(f"{self.root}/best.png", values)

        self.renderer.train()

    @torch.no_grad()
    def test_render(self, dataset: DataSet, fps: int = 20):
        """Test rendering (after training)"""

        assert not self.only_contour, "Not rendered yet."

        logs = {"inputs": [], "sketches": []}
        if self.use_contour:
            logs.update({"rgb_map": [], "disp_map": [], "acc_map": [], "curves": []})

        log_dir = f"{self.root}/test"
        for key in logs.keys():
            key_dir = os.path.join(log_dir, key)
            if not os.path.exists(key_dir):
                os.makedirs(key_dir)

        if self.save_svg:
            svg_dir = os.path.join(log_dir, "svgs")
            if not os.path.exists(svg_dir):
                os.makedirs(svg_dir)

        items = dataset.all_items
        all_rgbs, all_poses, all_rays = items["rgbs"], items["poses"], items["rays"]
        H, W, _ = dataset.get_HWF

        assert self.ckpt_dir is not None
        self.prepare(dataset, only_sq=False, train=False)

        self.renderer.eval()

        pbar = tqdm(range(len(dataset)), colour="blue")
        pbar.set_description("Test")
        for idx in pbar:
            rgbs, poses, rays = all_rgbs[idx], all_poses[idx], all_rays[idx : idx + 1]
            sketches, ret_dict = self.renderer(poses, rays, only_sq=False)

            if self.save_svg:
                self.renderer.save_svg(os.path.join(svg_dir, f"{idx}.svg"))

            rgbs = rgbs.cpu().numpy() * 255.0
            sketches = sketches.squeeze().clamp(0.0, 1.0).cpu().numpy() * 255.0

            logs["inputs"].append([idx, rgbs])
            logs["sketches"].append([idx, sketches])

            if self.use_contour:
                for k, v in ret_dict.items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        # print(f"! [Numerical Error] {k} contains nan or inf.")
                        v = v.nan_to_num()
                    v = v.repeat(1, 1, 3) if v.shape[-1] == 1 else v  # [H, W, 3]
                    v = v.clamp(0.0, 1.0).cpu().numpy() * 255.0
                    logs[k].append([idx, v])

        # save results
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for key, ls in logs.items():
            key_dir = os.path.join(log_dir, key)
            video = cv2.VideoWriter(f"{key_dir}/video.mp4", fourcc, fps, (W, H))
            for idx, value in ls:
                value = value.astype(np.uint8)
                imageio.imwrite(f"{key_dir}/{str(idx).zfill(3)}.png", value)
                video.write(cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
            video.release()
