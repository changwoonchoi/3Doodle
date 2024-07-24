import torch

EPS = 1.0e-8


def safe_pow(t: torch.Tensor, exp: float, eps: float = EPS):
    return t.clamp(eps).pow(exp)


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(torch.pow(x - y, 2)))


def cos_distance(x: torch.Tensor, y: torch.Tensor, eps: float = EPS, dim: int = None):
    if dim is None:
        return 1.0 - torch.cosine_similarity(x, y, eps=eps)
    else:
        return 1.0 - torch.cosine_similarity(x, y, dim=dim, eps=eps)


def euler2quat(angles=torch.Tensor) -> torch.Tensor:
    """euler angles to quaternion"""

    assert angles.shape[0] == 3, "angles should be [a, b, c] form"

    a, b, c = angles  # a = roll, b = pitch, c = yaw
    ca, sa = torch.cos(a * 0.5), torch.sin(a * 0.5)
    cb, sb = torch.cos(b * 0.5), torch.sin(b * 0.5)
    cc, sc = torch.cos(c * 0.5), torch.sin(c * 0.5)

    w = ca * cb * cc + sa * sb * sc
    x = sa * cb * cc - ca * sb * sc
    y = ca * sb * cc + sa * cb * sc
    z = ca * cb * sc - sa * sb * cc

    quaternion = torch.Tensor([w, x, y, z])

    return quaternion


def quat2euler(quat=torch.Tensor) -> torch.Tensor:
    """quaternion to euler angles"""

    assert quat.shape[0] == 4, "angles should be [w, x, y, z] form"

    w, x, y, z = quat

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if torch.abs(sinp) >= 1:
        pitch = torch.copysign(torch.pi / 2, sinp)
    else:
        pitch = torch.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    angles = torch.Tensor([roll, pitch, yaw])

    return angles


def euler2rot(angles: torch.Tensor) -> torch.Tensor:
    """euler angles to rotation matrix"""

    assert angles.shape[0] == 3, "angles should be [a, b, c] form"

    roll, pitch, yaw = angles

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    rot_z = torch.Tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rot_y = torch.Tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rot_x = torch.Tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    return rot_z @ (rot_y @ rot_x)


def quat2rot(quat: torch.Tensor) -> torch.Tensor:
    """quaternion to rotation matrix"""

    assert quat.shape[0] == 4, "angles should be [w, x, y, z] form"

    w, x, y, z = quat

    # calculate intermediate values for efficiency
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz = x * y, x * z
    yz = y * z

    # compute the rotation matrix
    rotmat = torch.Tensor(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (wy + xz)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (wx + yz), 1 - 2 * (xx + yy)],
        ]
    )

    return rotmat
