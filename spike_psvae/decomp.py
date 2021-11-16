import numpy as np
from tensorly.decomposition import parafac


def cumparafac(wfs, max_rank=100):
    errors = []
    for rank in range(max_rank):
        cptensor, _ = parafac(wfs, rank)
        rec = np.einsum("l,il,jl,kl->ijk", cptensor.weights, *cptensor.factors)
        mse = np.square(wfs - rec).sum(axis=(1, 2)).mean()
        errors.append(mse)
    return errors
