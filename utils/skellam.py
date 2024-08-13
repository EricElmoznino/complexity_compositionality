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
    x -= mean
    if x.dtype == torch.float:
        return_torch = True
        x, std = (
            x.detach().cpu().numpy(),
            std.detach().cpu().numpy(),
        )
    else:
        return_torch = False

    x, std = x / precision, std / precision
    x = np.round(x)
    mu = (std**2) / 2
    logpmf = skellam.logpmf(x, mu, mu)
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
    std = std / precision
    mu = (std**2) / 2
    samples = skellam.rvs(mu, mu, size=shape, random_state=random_state)
    samples = samples * precision + mean
    return samples
