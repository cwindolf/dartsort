import pytest
import torch
import torch.nn.functional as F

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
    assert torch.equal(dd - (K - 1), ii - jj)


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
    assert up_data.up_template_ix.shape == up_data.up_tt.shape

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
    if up_method == "keys3":
        atol = 1e-5
    elif deg == 1:
        atol = 5e-5
    elif deg == 2:
        atol = 2e-2
    else:
        assert False
    assert torch.allclose(yy_, yy_hat, atol=atol)


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
