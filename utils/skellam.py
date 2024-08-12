import numpy as np
import torch
from scipy.stats import skellam
from torch import FloatTensor


def approx_gaussian_logpmf(
    x: FloatTensor | np.ndarray,
    mean: FloatTensor | np.ndarray | float,
    std: FloatTensor | np.ndarray | float,
    precision: float = 1e-2,
) -> FloatTensor | np.ndarray:
    if x.dtype == torch.float32:
        return_torch = True
        x, mean, std = (
            x.detach().cpu().numpy(),
            mean.detach().cpu().numpy(),
            std.detach().cpu().numpy(),
        )
    else:
        return_torch = False

    breakpoint()
    x, mean, std = x / precision, mean / precision, std / precision
    var = std**2
    x, mean, var = np.round(x), np.round(mean), np.round(var)
    mu1 = (mean + var) / 2
    mu2 = mu1 - mean
    logpmf = skellam.logpmf(x, mu1, mu2)
    if return_torch:
        logpmf = torch.from_numpy(logpmf)
    return logpmf


def approx_gaussian_sample(
    mean: float | np.ndarray,
    std: float | np.ndarray,
    shape: tuple[int, ...],
    precision: float = 1e-2,
    random_state: np.random.RandomState | None = None,
):
    mean, std = mean / precision, std / precision
    var = std**2
    mean, var = np.round(mean), np.round(var)
    mu1 = (mean + var) / 2
    mu2 = mu1 - mean
    samples = skellam.rvs(mu1, mu2, size=shape, random_state=random_state)
    samples = samples * precision
    return samples
