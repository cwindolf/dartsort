import numpy as np
from tensorly.decomposition import parafac
from tqdm.auto import trange


def cumparafac(wfs, max_rank=100):
    errors = []
    for rank in trange(1, max_rank):
        weights, factors = parafac(wfs, rank)
        rec = np.einsum("l,il,jl,kl->ijk", weights, *factors)
        mse = np.square(wfs - rec).sum(axis=(1, 2)).mean()
        errors.append(mse)
    return np.array(errors)
