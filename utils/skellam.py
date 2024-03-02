from scipy.stats import skellam
from torch import FloatTensor


def approx_gaussian_logpmf(
    x: FloatTensor,
    mean: FloatTensor,
    std: FloatTensor,
    precision: float = 1e-2,
) -> float:
    x, mean, std = (
        x.detach().cpu().numpy(),
        mean.detach().cpu().numpy(),
        std.detach().cpu().numpy(),
    )
    x, mean, std = x / precision, mean / precision, std / precision
    var = std**2
    x, mean, var = x.round(), mean.round(), var.round()
    mu1 = (mean + var) / 2
    mu2 = mu1 - mean
    logpmf = skellam.logpmf(x, mu1, mu2)
    logpmf = logpmf.sum()
    return logpmf
