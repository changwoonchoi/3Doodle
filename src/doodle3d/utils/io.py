import argparse
import yaml

from typing import Dict, Any, Tuple, Union
from rich.pretty import pprint

from doodle3d.dataset.base import DataSet
from doodle3d.utils.misc import load_class


def load_dataset_testing(device: str, cfg: Dict[str, Any]) -> DataSet:
    class_type, params = f"doodle3d.{cfg['type']}", cfg["params"]
    params.update({"device": device, "mode": "test"})

    dataset = load_class(class_type, params)

    return dataset


def load_config_testing(
    device: str, args: argparse.Namespace, show: bool = True
) -> Tuple[Union[Dict[str, Any], str]]:
    assert args.ckpt_dir is not None
    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    exp_dir = cfg["method"]["exp_dir"]

    # update configs
    cfg["method"].update(
        {
            "device": device,
            "ckpt_dir": args.ckpt_dir,
            "grade": True,
            "clean": args.clean,
        }
    )
    # show values in configs
    if show:
        pprint(cfg)

    return cfg, exp_dir
