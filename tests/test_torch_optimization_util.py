import numpy as np
import torch
from dartsort.util.torch_optimization_util import batched_levenberg_marquardt
from torch import vmap
from torch.func import grad_and_value, hessian


def test_lm_basic():
    n = 10
    p = 11
    x0 = torch.ones((n, p)).double()

    # just checking we can minimize the norm
    def obj(z):
        return torch.linalg.norm(z)

    vgf = vmap(grad_and_value(obj))
    vhess = vmap(hessian(obj))

    assert not np.isclose(x0.numpy(), 0.0).any()
    x1, nsteps1 = batched_levenberg_marquardt(x0, vgf, vhess, max_steps=100)
    assert np.isclose(x1.numpy(), 0.0).all()
    # we won't converge here due to gradient condition, since
    # grad of norm is always 1
    assert (nsteps1 == 100).all()

    # gradient of norm is always 1 and this is a nice smooth problem,
    # so let's try to converge faster by setting params better
    assert not np.isclose(x0.numpy(), 0.0).any()
    x1, nsteps1 = batched_levenberg_marquardt(
        x0, vgf, vhess, convergence_g=1.01, min_scale=0.0
    )
    assert np.isclose(x1.numpy(), 0.0).all()
    assert (nsteps1 < 25).all()

    # just covering some branches not hit above
    assert not np.isclose(x0.numpy(), 0.0).any()
    x1, nsteps1 = batched_levenberg_marquardt(
        x0, vgf, vhess, scale_problem="none", tikhonov=0.01
    )
    assert np.isclose(x1.numpy(), 0.0).all()
