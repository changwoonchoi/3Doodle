import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--contour_img_size",
        nargs="+",
        type=int,
        default=[200, 200],
        help="resolution (h, w) to optimize contours",
    )
    parser.add_argument(
        "--curve_img_size",
        nargs="+",
        type=int,
        default=[400, 400],
        help="resolution (h, w) to optimize curves",
    )
    parser.add_argument(
        "--set_contour",
        action="store_true",
        help="optimize contours or not if the checkpoint is given",
    )

    parser.add_argument("-c", "--cuda", type=int, default=0, help="index of gpu")
    parser.add_argument(
        "-ep", "--exp_project", type=str, default="debug", help="project name"
    )
    parser.add_argument(
        "-eg", "--exp_group", type=str, default="debug", help="group name"
    )
    parser.add_argument(
        "-en", "--exp_name", type=str, default="debug", help="experiment name to save"
    )
    parser.add_argument(
        "-ev", "--exp_version", type=int, default=None, help="version of the experiment"
    )
    parser.add_argument(
        "-cf", "--config", type=str, default="nerf/lego.yaml", help="config file path"
    )
    parser.add_argument(
        "-ck", "--ckpt_dir", type=str, default=None, help="checkpoint path to load"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=0, help="seed to control randomness"
    )
    parser.add_argument("-f", "--fps", type=int, default=20, help="fps of videos")
    parser.add_argument(
        "-th", "--num_threads", type=int, default=None, help="number of threads to use"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="curve",
        choices=["contour", "curve", "all"],
        help="stage to learn",
    )
    parser.add_argument("--clean", action="store_true", help="clean strokes or not")
    parser.add_argument("--resume", action="store_true", help="continue or not")

    args = parser.parse_args()

    return args
