import torch

import pydiffvg

import signal
import os
import yaml
import time
import random
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from rich.pretty import pprint

from doodle3d.sketcher import Doodle
from doodle3d.dataset.base import DataSet
from doodle3d.utils.misc import ClearCache, signal_handler, load_class, update_dict
from doodle3d.utils.arguments import parse_args


def seed_all(seed: int = 1004):
    """Fix seed to control randomness"""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # current gpu seed
    torch.cuda.manual_seed_all(seed)  # All gpu seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(
    device: str, cfg: Dict[str, Any], resize: List[int]
) -> Dict[str, DataSet]:
    """Load datasets"""

    def update(params: Dict[str, Any], mode: str) -> Dict[str, Any]:
        params.update({"mode": mode})
        return params

    class_type, params = f"doodle3d.{cfg['type']}", cfg["params"]
    params.update({"device": device})
    # if "resize" not in params.keys():
    if resize is not None:
        params.update({"resize": resize})

    dataset = {
        "train": load_class(class_type, update(params, "train")),
        "eval": load_class(class_type, update(params, "val")),
    }

    return dataset


def load_config(
    device: str,
    version: int,
    args: argparse.Namespace,
    stages: str = "curve",
    show: bool = True,
) -> Tuple[Union[Dict[str, Any], str]]:
    """Load configs"""

    use_contour = "contour" in stages
    only_contour = use_contour and ("curve" not in stages)

    if args.config[:4] == "logs" or args.config[:6] == "./logs":
        cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    else:
        base_cfg = "all.yaml" if use_contour else "curve.yaml"
        cfg = yaml.load(open(f"./configs/{base_cfg}"), Loader=yaml.Loader)
        new_cfg = yaml.load(open(f"./{args.config}"), Loader=yaml.Loader)
        # recursively update configs
        cfg["data"] = update_dict(cfg["data"], new_cfg["data"])
        cfg["method"] = update_dict(cfg["method"], new_cfg["method"])

    if "exp_dir" in cfg["method"].keys() and args.resume:
        exp_dir = cfg["method"]["exp_dir"]
    else:
        exp_dir = f"{args.exp_project}/{args.exp_group}/{args.exp_name}_{version}"

    # update configs
    cfg["method"].update(
        {
            "device": device,
            "exp_dir": exp_dir,
            "ckpt_dir": args.ckpt_dir,
            "use_contour": use_contour,
            "only_contour": only_contour,
        }
    )
    # show values in configs
    if show:
        pprint(cfg)

    cfg_path = f"./logs/{exp_dir}/config.yaml"
    if not args.resume:
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

    return cfg, exp_dir


def count_version(args: argparse.Namespace) -> int:
    """Count version of the experiment"""

    version = args.exp_version
    basedir = f"./logs/{args.exp_project}/{args.exp_group}/{args.exp_name}"
    if version is None:
        version = 0
        exp_dir = f"{basedir}_{version}"
        while os.path.exists(exp_dir):
            version += 1
            exp_dir = f"{basedir}_{version}"
    else:
        exp_dir = f"{basedir}_{version}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    return version


def main():
    """Main function to train"""

    args = parse_args()

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    seed_all(args.seed)

    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(torch.device(device))

    stages = "contour_curve" if args.stages == "all" else args.stages
    set_contour = (
        True if ("contour" in stages and (args.ckpt_dir is None)) else args.set_contour
    )

    version = count_version(args)
    config, exp_dir = load_config(device, version, args, stages, show=True)

    # print path of directory to save
    print(f"[exp_dir] ./logs/{exp_dir}")

    start = time.time()

    sketcher = Doodle(**config["method"])

    if ("contour" in stages) and set_contour:
        print(f"learning contours started")
        dataset = load_dataset(device, config["data"], args.contour_img_size)
        sketcher.learn(dataset["train"], dataset["eval"], only_sq=True)
        del dataset
    if "curve" in stages:
        print(f"learning curves started")
        dataset = load_dataset(device, config["data"], args.curve_img_size)
        sketcher.learn(dataset["train"], dataset["eval"], only_sq=False)

    else:
        times = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        print(f"Finished drawing! [time taken] {times}")

    del dataset
    del sketcher


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    with ClearCache():
        main()
