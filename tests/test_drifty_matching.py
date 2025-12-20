import pytest
import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import correlate

from dartsort.peel.matching_util import drifty


@pytest.fixture
def hires_templates():
    pass


@pytest.fixture
def hires_pconv(hires_templates):
    pass


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

    conv_len = 2 * t - 1

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

    tconv0 = np.zeros((rank, rank, up, conv_len), dtype=np.float32)
    tconv00 = np.zeros((rank, rank, up, conv_len), dtype=np.float32)
    for p in range(rank):
        for q in range(rank):
            for u in range(up):
                tconv0[p, q, u] = correlate(
                    temporal[p], temporal_up[q, u], mode="full", method="direct"
                )[::-1]
                tconv00[p, q, u] = correlate(
                    temporal[p][::-1],
                    temporal_up[q, u][::-1],
                    mode="full",
                    method="direct",
                )

    # different order in sums helps figure out what numerical tolerance is appropriate
    tconv_atol = np.abs(tconv0 - tconv00).max() + np.finfo(np.float32).tiny
    assert tconv_atol < 1e-5
    np.testing.assert_allclose(tconv0, tconv00, atol=tconv_atol)

    tconv1_ = drifty.shared_temporal_pconv(temporal_, temporal_up_)
    tconv1 = tconv1_.numpy(force=True)

    # doubling tolerance because GPUs are GPUs
    np.testing.assert_allclose(tconv0, tconv1, strict=True, atol=2 * tconv_atol)

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
    np.testing.assert_allclose(
        full_pconv0, full_pconv1, strict=True, atol=2 * max(pconv_atol, gpu_pconv_atol)
    )
    np.testing.assert_allclose(
        full_pconv0, full_pconv2, strict=True, atol=2 * max(pconv_atol, gpu_pconv_atol)
    )


@pytest.mark.parametrize("deg", [1, 2])
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
    if deg == 2:
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
    if up_method in ("keys3", "keys4"):
        atol = 1e-5
    elif deg == 1:
        atol = 1e-4
    elif deg == 2:
        atol = 2e-2
    else:
        assert False

    assert torch.allclose(yy_, yy_hat, atol=atol)


def test_direct_vs_keys():
    # should agree when templates are parabolas, yea?
    pass


@pytest.mark.parametrize("up", [1, 2, 4])
def test_shared_compression(up):
    # for high ish rank, should be able to compress the simulator templates well
    pass


@pytest.mark.parametrize("up", [1, 2, 4])
def test_drift_interpolation(up):
    pass


@pytest.mark.parametrize("up", [1, 2, 4])
def test_static_pconv(up):
    pass


@pytest.mark.parametrize("up", [1, 2, 4])
def test_pitch_shift_matching_nearest(up):
    """With nearest interpolation and pitch shift motion, residual can be 0 with no compression.

    Residual should be near 0 with thin plates, and shared basis may push it away from 0 a bit.
    With refractory recording, we can get the exact upsampling indices and scalings. With no noise,
    scores should be exactly the (compressed) template norms squared.
    """
    pass
