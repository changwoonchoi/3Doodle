import torch

import pydiffvg

import signal

from doodle3d.sketcher import Doodle
from doodle3d.utils.misc import signal_handler
from doodle3d.utils.arguments import parse_args
from doodle3d.utils.io import load_config_testing, load_dataset_testing


def main() -> None:  # XXX fix this
    """Main function to render test view images."""

    args = parse_args()

    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

    config, exp_dir = load_config_testing(device, args, show=True)
    exp_project, exp_group, exp_name = exp_dir.split("/")

    # print path of directory to save
    print(f"[exp_dir] ./logs/{exp_dir}")

    dataset = load_dataset_testing(device, config["data"])
    sketcher = Doodle(**config["method"])

    sketcher.test_render(dataset=dataset, fps=args.fps)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
