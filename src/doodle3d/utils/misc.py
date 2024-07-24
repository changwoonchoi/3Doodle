import torch
import torch.nn.functional as F
from torchvision import transforms

import os
import gc
import sys
import signal
import imp
import random
import math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

SFM_TYPES = ["sfm", "edge-sfm", "random"]
RENDER_TYPES = [
    "sigma_ISCO",
    "contour_ISCO",
    "contour_3doodle",
    "contour_3doodle_adaptive",
]
MODEL_TYPES = [
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px",
]
OPTIM_TYPES = ["RGBA", "RGB", "Alpha", "Default"]
PROJ_TYPES = ["ortho", "persp"]
RAND_TYPES = ["hemisphere", "bbox"]
DIST_TYPES = ["L1", "L2", "Cos", "LPIPS"]
EDGE_TYPES = ["canny", "pidinet"]
REG_TYPES = ["mean", "sum"]


class ClearCache:
    def __enter__(self):
        for obj in gc.get_objects():
            try:
                del obj
            except:
                pass
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj in gc.get_objects():
            try:
                del obj
            except:
                pass
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


def signal_handler(sig, frame):
    signal.signal(sig, signal.SIG_IGN)
    sys.exit(0)


def conditional_decorator(dec, condition: bool):
    def decorator(func):
        if not condition:
            return func
        return dec(func)

    return decorator


def load_class(type_str: str, params: Dict[str, Any]):
    paths = type_str.split(".")
    root, main, subs = paths[0], paths[-1], []
    if len(paths) > 2:
        subs = paths[1:-1]

    f, fname, desc = imp.find_module(root)
    module = imp.load_module(root, f, fname, desc)
    for sub in subs:
        path = os.path.dirname(fname) if os.path.isfile(fname) else fname
        f, fname, desc = imp.find_module(sub, [path])
        try:
            module = imp.load_module(" ".join(paths[:-1]), f, fname, desc)
        finally:
            if f:
                f.close()

    return getattr(module, main)(**params)


def update_dict(parent: Dict, child: Dict) -> Dict:
    for key, value in child.items():
        try:
            if type(value) == dict:
                value = update_dict(parent[key], value)
        except KeyError:
            pass
        parent[key] = value

    return parent


def get_augment_trans(
    affine: bool = False,
    fill: int = 0,
    p: float = 1.0,
    distortion_scale: float = 0.5,
    size: int = 224,
    scale: tuple[float] = (0.8, 0.8),
    ratio: tuple[float] = (1.0, 1.0),
):
    augments = []
    if affine:
        augments.append(
            transforms.RandomPerspective(
                fill=fill, p=p, distortion_scale=distortion_scale
            )
        )
        augments.append(transforms.RandomResizedCrop(size, scale=scale, ratio=(ratio)))
    augments.append(
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )
    )
    augment_trans = transforms.Compose(augments)

    return augment_trans


def stats_to_tensor(
    mean: List[int], std: List[int], device: str
) -> Tuple[torch.Tensor]:
    mean = torch.tensor(mean).to(device).view(1, -1, 1, 1)
    std = torch.tensor(std).to(device).view(1, -1, 1, 1)

    return mean, std


def get_rand_fn(hemisphere: bool = True, z_nonneg: bool = True):
    def rand_from_hemisphere(
        radius: float = 4.0, device: str = "cuda:0"
    ) -> torch.Tensor:
        theta = torch.deg2rad(torch.Tensor([random.random() * 360 - 180]))
        phi = torch.deg2rad(torch.Tensor([random.random() * (90)]))

        cos_theta, sin_theta = torch.cos(theta).item(), torch.sin(theta).item()
        cos_phi, sin_phi = torch.cos(phi).item(), torch.sin(phi).item()

        coord = torch.Tensor(
            [
                (radius * cos_phi) * cos_theta,
                (radius * cos_phi) * sin_theta,
                radius * sin_phi,
            ]
        ).to(device)

        return coord

    def rand_from_bbox_up(xyz: List[float], device: str = "cuda:0") -> torch.Tensor:
        assert len(xyz) == 3, "Invalid format."

        xvalue = random.uniform(-1, 1) * xyz[0]
        yvalue = random.uniform(-1, 1) * xyz[1]
        zvalue = random.uniform(0, 1) * xyz[2]

        point = torch.Tensor([xvalue, yvalue, zvalue]).to(device)

        return point

    def rand_from_bbox(xyz: List[float], device: str = "cuda:0") -> torch.Tensor:
        assert len(xyz) == 3, "Invalid format."

        xvalue = random.uniform(-1, 1) * xyz[0]
        yvalue = random.uniform(-1, 1) * xyz[1]
        zvalue = random.uniform(-1, 1) * xyz[2]

        point = torch.Tensor([xvalue, yvalue, zvalue]).to(device)

        return point

    if hemisphere:
        return rand_from_hemisphere
    else:
        return rand_from_bbox_up if z_nonneg else rand_from_bbox


def get_stroke_size(x: torch.Tensor) -> torch.Tensor:
    num_pts, _ = x.shape

    dists = []
    for i in range(num_pts):
        x1 = x[i, :].repeat(num_pts, 1)
        dist = F.pairwise_distance(x, x1).max()
        dists.append(dist)
    size = torch.concat(dist).max()

    return size


def get_mean_dist(x: torch.Tensor) -> torch.Tensor:
    num_pts, _ = x.shape

    dists = []
    for i in range(num_pts):
        x1 = x[i][None, ...].repeat(4, 1)
        dist = x - x1
        dists.append(dist)
    means = torch.mean(torch.stack(dists, dim=0), dim=0)
    mean_dist = torch.linalg.norm(means)

    return mean_dist


def blender2world(
    R: torch.Tensor, T: torch.Tensor, device: str = "cuda:0"
) -> torch.Tensor:
    """transfer coordinates of projection (blender -> world)"""

    proj = torch.eye(4)
    R[1:, ...] *= -1
    T[1:, ...] *= -1
    proj[:3, :3], proj[:3, -1:] = R, T

    return proj.to(device)


def blender2openGL(
    R: torch.Tensor, T: torch.Tensor, device: str = "cuda:0"
) -> torch.Tensor:
    """transfer coordinates of projection (blender -> OpenGL)"""

    proj = torch.eye(4)

    # | a  b  c  tx |       | a  c  -b  tx |
    # | d  e  f  ty |  -->  | g  i  -h  tz |
    # | g  h  i  tz |       | d  f  -e  ty |
    # | 0  0  0  1  |       | 0  0  0   1  |

    R[1, ...], R[2, ...] = R[2, ...], R[1, ...]
    R[..., 1], R[..., 2] = R[..., 2], -R[..., 1]
    T[1, ...], T[2, ...] = T[2, ...], -T[1, ...]

    proj[:3, :3], proj[:3, -1:] = R, T

    return proj.to(device)


def colmap2blender(points: torch.Tensor) -> torch.Tensor:
    """convert coordinates (colmap -> blender)
    (x_blender, y_blender, z_blender) = (x_colmap, -z_colmap, y_colmap)

    Args:
        points (torch.Tensor): [N, 3]
    """

    y, z = points[..., 1], points[..., 2]
    points[..., 1] = -z
    points[..., 2] = y

    return points


def get_rand_tensor(
    size: List[int],
    min: float = 0.0,
    max: float = 2.0,
    eps: float = 1e-6,
    open_interval: bool = True,
):
    """create random tensors in open interval (min, max)"""

    tensor = (max - min) * torch.rand(size) + min
    if open_interval:
        tensor = torch.max(tensor, torch.ones_like(tensor) * min + eps)

    return tensor


def get_euler_angle(randomly=True) -> torch.Tensor:
    """randomly set euler angle"""

    if randomly:
        yaw = random.uniform(0, 2 * torch.pi)
        pitch = random.uniform(0, torch.pi)
        roll = random.uniform(0, 2 * torch.pi)
        angle = torch.Tensor([yaw, pitch, roll])
    else:
        angle = torch.zeros([3])

    return angle


def line_distance(l1: List[torch.Tensor], l2: List[torch.Tensor]) -> torch.Tensor:
    """compute line distance in 3D space."""

    assert len(l1) == 2 and len(l2) == 2

    s1, e1 = l1
    s2, e2 = l2

    u = e1 - s1
    v = e2 - s2
    w = s1 - s2

    cross_uv = torch.cross(u, v)
    cross_uw = torch.cross(u, w)
    cross_vw = torch.cross(v, w)

    a = torch.dot(u, u)
    b = torch.dot(u, v)
    c = torch.dot(v, v)
    d = torch.dot(u, w)
    e = torch.dot(v, w)
    denom = a * c - b * b

    if torch.norm(cross_uv) == 0:
        dist = torch.norm(cross_uw) / torch.norm(u)
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        closest_point_on_line1 = s1 + s * u
        closest_point_on_line2 = s2 + t * v
        dist = torch.norm(closest_point_on_line1 - closest_point_on_line2)

    return dist


def rand_on_circle(
    p1: torch.Tensor, p2: torch.Tensor, num_points: int = 2
) -> List[torch.Tensor]:
    """randomly select poins from the circle"""

    radius_vector = p2 - p1
    radius = torch.norm(radius_vector)

    points = []
    for _ in range(num_points):
        random_vec = torch.randn(3, device=p1.device)
        tan1 = torch.cross(radius_vector, random_vec)
        tan1 /= torch.norm(tan1)  # Normalize

        tan2 = torch.cross(radius_vector, tan1)
        tan2 /= torch.norm(tan2)  # Normalize

        angle = 2 * torch.pi * torch.rand(1, device=p1.device)
        point = p1 + radius * (torch.cos(angle) * tan1 + torch.sin(angle) * tan2)
        points.append(point)

    return points


def rand_on_line(
    p1: torch.Tensor, p2: torch.Tensor, num_points: int = 2
) -> List[torch.Tensor]:
    """randomly select poins from the line"""

    points = []
    for _ in range(num_points):
        t = torch.rand(1, device=p1.device).item()
        point = (1 - t) * p1 + t * p2
        points.append(point)

    return points


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


@dataclass(frozen=True)
class HWF:
    width: float
    height: float
    focal: float


def pose_to_rays(
    pose: torch.FloatTensor,
    width: int,
    height: int,
    focal: int,
    device: torch.DeviceObjType,
) -> torch.FloatTensor:
    pose = pose.to(device)
    w_range, h_range = torch.arange(width, device=device), torch.arange(
        height, device=device
    )
    xs, ys = torch.meshgrid(w_range, h_range, indexing="xy")

    # blender projection
    dirs = torch.stack(
        [
            (xs - width / 2) / focal,
            -(ys - height / 2) / focal,
            -torch.ones_like(xs, device=device),
        ],
        dim=-1,
    )
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)  # [H, W, 3]
    rays_o = pose[:3, -1].expand(rays_d.shape)  # [H, W, 3]

    return torch.concat([rays_o, rays_d], dim=-1)  # [H, W, 6]
