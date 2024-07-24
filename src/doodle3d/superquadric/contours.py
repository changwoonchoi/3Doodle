import torch

from doodle3d.utils.misc import conditional_decorator


def sigma_ISCO(superquadric_val, gamma):
    """ISCO-style sigma value of superquadric."""
    sigma = torch.sigmoid(gamma * (1 - superquadric_val))

    return sigma


def contour_ISCO(superquadric_val, gamma, bias, scale, eps=0.01):
    """Contour of superquadric with gradient of rendered image of ISCO-style sigma."""

    sigma = scale * torch.sigmoid(
        1.0 / (gamma * (1 - superquadric_val) ** 2 + eps)
        - gamma * (1 - superquadric_val) ** 2
        - bias
    )

    return sigma


def contour_3doodle(superquadric_val, normal, viewdirs, gamma, bias, scale, beta):
    """Our proposed superquadric contour function."""

    surface_term = torch.sigmoid(
        1.0 / (gamma * (1 - superquadric_val) ** 2 + 0.01)
        - gamma * (1 - superquadric_val) ** 2
        - bias
    )
    # attenuation = gamma_2 * (1 - torch.einsum('ijk,ik->ij', normal, viewdirs) ** beta)
    if len(normal.shape) == 3 and len(viewdirs.shape) == 2:
        einsum_str = "ijk,ik->ij"
    elif len(normal.shape) == 2 and len(viewdirs.shape) == 2:
        einsum_str = "ij,ij->i"
    else:
        raise ValueError
    attenuation = (1 - torch.abs(torch.einsum(einsum_str, normal, viewdirs))) ** beta
    sigma = scale * attenuation * surface_term

    return sigma


def contour_3doodle_adaptive(
    alpha,
    epsilon,
    superquadric_val,
    normal,
    viewdirs,
    beta,
    adaptive_param_grad,
    gamma_min,
    gamma_max,
    bias_min,
    bias_max,
    scale_min,
    scale_max,
):
    """Our proposed superquadric contour function.
    Adaptive line width respective to shape and scale.
    """

    @conditional_decorator(torch.no_grad(), not adaptive_param_grad)
    def adaptive_gamma(alpha, epsilon, gamma_1_min, gamma_1_max):
        gamma_scale = gamma_1_min + (gamma_1_max - gamma_1_min) * (alpha - 0.1) / 0.9

        epsilon_mask = epsilon > 1
        gamma_shape = gamma_1_min + (gamma_1_max - gamma_1_min) * (epsilon - 0.1) / 0.9
        gamma_shape[epsilon_mask] = gamma_1_max

        gamma = torch.min(gamma_scale, gamma_shape)
        return gamma

    @conditional_decorator(torch.no_grad(), not adaptive_param_grad)
    def adaptive_bias(alpha, epsilon, bias_min, bias_max):
        bias_scale = bias_min + (bias_max - bias_min) * (alpha - 0.1) / 0.9

        epsilon_mask = epsilon > 1
        bias_shape = bias_min + (bias_max - bias_min) * (epsilon - 0.1) / 0.9
        bias_shape[epsilon_mask] = bias_max

        bias = torch.min(bias_scale, bias_shape)
        return bias

    @conditional_decorator(torch.no_grad(), not adaptive_param_grad)
    def adaptive_scale(alpha, epsilon, scale_min, scale_max):
        scale_mask = alpha > 0.3
        scale_scale = scale_max - (scale_max - scale_min) * (alpha - 0.1) / 0.2
        scale_scale[scale_mask] = scale_min

        epsilon_mask = epsilon > 0.3
        scale_shape = scale_max - (scale_max - scale_min) * (epsilon - 0.1) / 0.2
        scale_shape[epsilon_mask] = scale_min

        scale = torch.max(scale_scale, scale_shape)
        return scale

    gamma = adaptive_gamma(alpha, epsilon, gamma_min, gamma_max)
    bias = adaptive_bias(alpha, epsilon, bias_min, bias_max)
    scale = adaptive_scale(alpha, epsilon, scale_min, scale_max)

    surface_term = torch.sigmoid(
        1.0 / (gamma * (1 - superquadric_val) ** 2 + 0.01)
        - gamma * (1 - superquadric_val) ** 2
        - bias
    )

    if len(normal.shape) == 3 and len(viewdirs.shape) == 2:
        einsum_str = "ijk,ik->ij"
    elif len(normal.shape) == 2 and len(viewdirs.shape) == 2:
        einsum_str = "ij,ij->i"
    else:
        raise ValueError

    attenuation = (1 - torch.abs(torch.einsum(einsum_str, normal, viewdirs))) ** beta
    sigma = scale * attenuation * surface_term

    return sigma
