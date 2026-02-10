import pytest
import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import correlate

from dartsort.peel.matching_util import drifty
from dartsort.util.testing_util import matching_debug_util


test_K = 11
test_template_nc = [1, 4]


@pytest.mark.parametrize("up", [1, 2, 4, 16])
@pytest.mark.parametrize("K", [1, 2, 5])
def test_shared_temporal_pconv(K, up):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp = torch.eye(K, device=dev)
    up_temp = temp[:, None, :].broadcast_to(K, up, K)
    pconv = drifty.shared_temporal_pconv(temp, up_temp)
    assert pconv.shape == (K, K, up, 2 * K - 1)
    assert pconv.sum() == K * K * up
    ii, jj, uu, dd = pconv.cpu().nonzero(as_tuple=True)
    assert torch.equal(dd - (K - 1), jj - ii)


@pytest.mark.parametrize("t", [11])
@pytest.mark.parametrize("rank", [1, 5])
@pytest.mark.parametrize("nc", [1, 5, -1])
@pytest.mark.parametrize("up", [1, 2, 4, 16])
@pytest.mark.parametrize("K", [1, 2, 5])
def test_full_shared_pconv(K, up, nc, rank, t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rg = np.random.default_rng(0)

    ortho = nc == -1
    if ortho:
        # orthogonal mode, each template on its own channel
        nc = K

    if ortho:
        spatial_sing = np.zeros((K, rank, nc), dtype=np.float32)
        ix = np.arange(K)
        spatial_sing[ix, :, ix] = rg.normal(size=(K, rank))
    else:
        spatial_sing = rg.normal(size=(K, rank, nc)).astype(np.float32)
    temporal = rg.normal(size=(rank, t)).astype(np.float32)
    temporal_up = rg.normal(size=(rank, up, t)).astype(np.float32)

    spatial_sing_ = torch.asarray(spatial_sing, device=device)
    temporal_ = torch.asarray(temporal, device=device)
    temporal_up_ = torch.asarray(temporal_up, device=device)

    tconv0, tconv00 = matching_debug_util.reference_shared_temporal_convolution(
        temporal, temporal_up
    )

    # different order in sums helps figure out what numerical tolerance is appropriate
    tconv_atol = np.abs(tconv0 - tconv00).max() + np.finfo(np.float32).tiny
    assert tconv_atol < 1e-5
    np.testing.assert_allclose(tconv0, tconv00, atol=tconv_atol)

    tconv1_ = drifty.shared_temporal_pconv(temporal_, temporal_up_)
    tconv1 = tconv1_.numpy(force=True)

    # doubling tolerance because GPUs are GPUs
    assert tconv0.shape == tconv1.shape
    assert tconv0.dtype == tconv1.dtype
    np.testing.assert_allclose(tconv0, tconv1, atol=2 * tconv_atol)

    # similarly pick a rounding error here
    full_pconv0 = np.einsum("ipc,pqul,jqc->ijul", spatial_sing, tconv0, spatial_sing)
    full_pconv00 = np.einsum(
        "ipc,pqul,jqc->ijul", spatial_sing[::-1], tconv00, spatial_sing
    )
    full_pconv00 = full_pconv00[::-1]
    pconv_atol = np.abs(full_pconv0 - full_pconv00).max() + np.finfo(np.float32).tiny
    assert pconv_atol < 1e-4
    np.testing.assert_allclose(full_pconv0, full_pconv00, atol=pconv_atol)

    tconv0_ = torch.asarray(tconv0, device=device)
    full_pconv000_ = torch.einsum(
        "ipc,pqul,jqc->ijul", torch.flip(spatial_sing_, (0,)), tconv0_, spatial_sing_
    )
    full_pconv000 = full_pconv000_.numpy(force=True)[::-1]
    gpu_pconv_atol = (
        np.abs(full_pconv000 - full_pconv0).max() + np.finfo(np.float32).tiny
    )
    assert gpu_pconv_atol < 1e-4
    np.testing.assert_allclose(full_pconv000, full_pconv0, atol=gpu_pconv_atol)

    full_pconv1_ = torch.einsum(
        "ipc,pqul,jqc->ijul", spatial_sing_, tconv1_, spatial_sing_
    )
    full_pconv1 = full_pconv1_.numpy(force=True)
    full_pconv2_ = drifty.full_shared_pconv(
        tconv1_, spatial_sing_, batch_size=max(2, K // 2)
    )
    full_pconv2 = full_pconv2_.numpy(force=True)

    if ortho:
        targ = np.eye(K, dtype=np.bool_)[:, :, None, None]
        targ = np.broadcast_to(targ, full_pconv0.shape)
        np.testing.assert_array_equal(full_pconv0 != 0, targ)
    assert full_pconv0.shape == full_pconv1.shape
    assert full_pconv0.dtype == full_pconv1.dtype
    np.testing.assert_allclose(
        full_pconv0, full_pconv1, atol=2 * max(pconv_atol, gpu_pconv_atol)
    )
    assert full_pconv0.shape == full_pconv2.shape
    assert full_pconv0.dtype == full_pconv2.dtype
    np.testing.assert_allclose(
        full_pconv0, full_pconv2, atol=2 * max(pconv_atol, gpu_pconv_atol)
    )


@pytest.mark.parametrize("deg", [1, 2, 3])
@pytest.mark.parametrize("radius", [8])
@pytest.mark.parametrize("up", [2, 4, 16])
@pytest.mark.parametrize("up_method", ["interpolation", "keys3", "keys4"])
def test_interp_upsampling(up_method, up, radius, deg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test keys and thinplate recover a cubic function correctly
    up_data = drifty.get_interp_upsampling_data(
        up_factor=up, up_method=up_method, interp_up_radius=radius, device=device
    )

    # check domains are as expected
    if up_method == "keys3":
        radius = 2
    elif up_method == "keys4":
        radius = 3
    assert torch.equal(up_data.objective_tt.cpu(), torch.arange(-radius, radius + 1))
    assert torch.equal(up_data.up_tt.cpu(), torch.arange(-up // 2, up // 2 + 1) / up)
    assert torch.allclose(up_data.up_tt.diff(), torch.tensor(1.0 / up))
    assert up_data.up_ix.shape == up_data.up_tt.shape

    # test fns
    zz = up_data.objective_tt
    zz_ = up_data.up_tt
    if deg == 3:
        yy = -3.0 * zz**3 - 10.0 * zz**2 + 8.0
        yy_ = -3.0 * zz_**3 - 10.0 * zz_**2 + 8.0
    elif deg == 2:
        yy = 4.0 * zz**2 - 6.0 * zz - 3.0
        yy_ = 4.0 * zz_**2 - 6.0 * zz_ - 3.0
    elif deg == 1:
        yy = zz + 2.0
        yy_ = zz_ + 2.0
    else:
        assert False

    # upsampled reconstruction
    if up_data.zpad:
        yy = F.pad(yy, (0, up_data.zpad))
    yy_hat = yy @ up_data.interpolator

    # keys is better than thin plates for these test fns
    if up_method == "keys3" and deg == 3:
        # keys3 not good enough! use keys4.
        atol = 0.3
    elif up_method == "keys3":
        atol = 1e-6
    elif up_method == "keys4":
        atol = 1e-4
    elif deg == 1:
        atol = 1e-4
    elif deg == 2:
        atol = 2e-2
    elif deg == 3:
        # kind of indicates that this is a bad method here.
        atol = 0.1
    else:
        assert False

    np.testing.assert_allclose(
        yy_.numpy(force=True), yy_hat.numpy(force=True), atol=atol
    )


@pytest.mark.parametrize("up", [1, 2, 4])
def test_pitch_shift_matching_nearest(up):
    """With nearest interpolation and pitch shift motion, residual can be 0 with no compression.

    Residual should be near 0 with thin plates, and shared basis may push it away from 0 a bit.
    With refractory recording, we can get the exact upsampling indices and scalings. With no noise,
    scores should be exactly the (compressed) template norms squared.
    """
    pass
