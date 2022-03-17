"""The home of tried and true methods. Right now we have one.
"""
import numpy as np
import pywt
import torch
from scipy import sparse
from scipy.stats import zscore, norm
from scipy.interpolate import interp1d, RectBivariateSpline
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm.auto import trange, tqdm

from .ibme_fast_raster import raster as _raster


def kronsolve(D, robust_sigma=0):
    T, T_ = D.shape
    assert T == T_

    if robust_sigma > 0:
        Dz = zscore(D, axis=1)
        S = (np.abs(Dz) < robust_sigma).astype(D.dtype)

    _1 = np.ones((T, 1))
    Id = sparse.identity(T)
    kron = sparse.kron(Id, _1).tocsr() - sparse.kron(_1, Id).tocsr()

    if robust_sigma <= 0:
        p, *_ = sparse.linalg.lsqr(kron, D.ravel())
    else:
        dvS = sparse.diags(S.ravel())
        p, *_ = sparse.linalg.lsqr(dvS @ kron, dvS @ D.ravel())

    return p


def decentrigid(raster, robust_sigma=0, batch_size=1, step_size=1, disp=400):
    # this is not implemented but would be done by the stride.
    assert step_size == 1
    T = raster.shape[1]
    possible_displacement = np.arange(-disp, disp + step_size, step_size)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # print("possdisp", possible_displacement.shape)

    with torch.no_grad():
        raster = torch.as_tensor(raster.T, dtype=torch.float32, device=device)
        raster = raster[:, None, :]
        c2d = torch.nn.Conv2d(
            in_channels=1,
            out_channels=T,
            kernel_size=[1, raster.shape[-1]],
            stride=1,
            padding=[0, possible_displacement.size // 2],
            bias=False,
        ).to(device)
        c2d.weight[:, 0] = raster
        displacement = np.empty((T, T))
        for i in trange(T // batch_size):
            res = c2d(raster[i * batch_size : (i + 1) * batch_size, None])[
                :, :, 0, :
            ]
            # print("res shape", res.shape)
            res = res.argmax(2)
            displacement[
                i * batch_size : (i + 1) * batch_size
            ] = possible_displacement[res.cpu()]
            del res

    p = kronsolve(displacement, robust_sigma=robust_sigma)

    return displacement, p


def register_rigid(
    amps,
    depths,
    times,
    robust_sigma=0,
    batch_size=1,
    step_size=1,
    disp=400,
    denoise_sigma=0.1,
    destripe=False,
):
    depths_reg = depths
    raster, dd, tt = fast_raster(
        amps, depths_reg, times, sigma=denoise_sigma, destripe=destripe
    )

    D, p = decentrigid(
        raster,
        robust_sigma=robust_sigma,
        batch_size=batch_size,
        step_size=step_size,
        disp=disp,
    )
    warps = interp1d(tt + 0.5, p, fill_value="extrapolate")(times)
    depths_reg = depths_reg - warps
    depths_reg -= depths_reg.min()
    return depths_reg, p


def compose_shifts_in_orig_domain(orig_shift, new_shift):
    """
    Context: we are in the middle of an iterative algorithm which
    computes displacements. Previously, we had computed some DxT
    displacement matrix `orig_shift`, giving a displacement at all
    depths and times. Then, we used that to compute a new registered
    raster, and we got a new displacement estimate from that new
    raster, `new_shift`. The problem is, that new raster may have a
    different depth domain than the original -- so, how do we combine
    these?

    Like so: if (x, t) is a point in the original domain, then think of
    `orig_shift` as a function f(x, t) giving the displacement at depth
    x and time t. Similarly, write `new_shift` as g(x', t), where x' is
    a point in the domain after it has been displaced by f, i.e.,
    x' = f(x, t) + x. Then we want to know, how much is a point (x,t)
    displaced by applying both f and g?

    The answer:
        (x, t) -> (f(x, t) + g(f(x, t) + x, t), t).
    So, this function returns h(x, t) = f(x, t) + g(f(x, t) + x, t)
    """
    D, T = orig_shift.shape
    D_, T_ = new_shift.shape
    assert T == T_

    orig_depth_domain = np.arange(D, dtype=float)
    g_domain = np.arange(D_, dtype=float)
    time_domain = np.arange(T, dtype=float)

    x_plus_f = orig_depth_domain[:, None] + orig_shift

    g_lerp = RectBivariateSpline(
        g_domain,
        time_domain,
        new_shift,
        kx=1,
        ky=1,
    )
    h = g_lerp(x_plus_f.ravel(), np.tile(time_domain, D), grid=False)

    return orig_shift + h.reshape(orig_shift.shape)


def warp_nonrigid(depths, times, dispmap, depth_domain=None, time_domain=None):
    if depth_domain is None:
        depth_domain = np.arange(int(np.floor(depths.max())) + 1)
    if time_domain is None:
        time_domain = np.arange(int(np.floor(times.max())) + 1)

    lerp = RectBivariateSpline(
        depth_domain,
        time_domain,
        dispmap,
        kx=1,
        ky=1,
    )

    return depths - lerp(depths, times, grid=False)


def register_nonrigid(
    amps,
    depths,
    times,
    robust_sigma=0,
    batch_size=1,
    step_size=1,
    rigid_disp=400,
    disp=400,
    denoise_sigma=0.1,
    destripe=False,
    n_windows=10,
    n_iter=1,
    widthmul=0.5,
):
    origmean = depths.mean()

    # set origin to min z
    depths = depths - depths.min()

    # initialize displacement map
    D = 1 + int(np.floor(depths.max()))
    T = int(np.floor(times.max())) + 1
    total_shift = np.zeros((D, T))

    # first pass of rigid registration
    depths, p = register_rigid(
        amps,
        depths,
        times,
        robust_sigma=robust_sigma,
        batch_size=batch_size,
        step_size=step_size,
        disp=rigid_disp,
        denoise_sigma=denoise_sigma,
        destripe=destripe,
    )
    total_shift[:, :] = p[None, :]

    pyramid = True
    if not isinstance(n_windows, list):
        n_windows = [n_windows] * n_iter
        pyramid = False

    for nwin in tqdm(n_windows):
        raster, dd, tt = fast_raster(
            amps, depths, times, sigma=denoise_sigma, destripe=destripe
        )
        D, T = raster.shape

        # gaussian windows
        windows = np.empty((nwin, D))
        space = D // (nwin + 1)
        locs = np.linspace(space, D - space, nwin)
        scale = widthmul * D / nwin
        scale = scale - scale % 2
        for k, loc in enumerate(locs):
            windows[k, :] = norm.pdf(np.arange(D), loc=loc, scale=scale)
        windows /= windows.sum(axis=0, keepdims=True)

        # estimate each window's displacement
        ps = np.empty((nwin, T))
        for k, window in enumerate(tqdm(windows, desc="windows")):
            D, p = decentrigid(
                (window[:, None] * raster).astype(np.float32).copy(),
                robust_sigma=robust_sigma,
                batch_size=batch_size,
                step_size=step_size,
                disp=min(scale, disp) if pyramid else disp,
            )
            ps[k] = p

        # warp depths
        dispmap = windows.T @ ps
        depths = warp_nonrigid(
            depths, times, dispmap, depth_domain=dd, time_domain=tt
        )
        depths -= depths.min()
        raster, dd, tt = fast_raster(
            amps, depths, times, sigma=denoise_sigma, destripe=destripe
        )

        # update displacement map
        total_shift[:, :] = compose_shifts_in_orig_domain(total_shift, dispmap)

    # back to original coordinates
    depths -= (depths.mean() - origmean)
    total_shift -= total_shift.mean()

    return depths, total_shift


# -- code for making rasters


def fast_raster(
    amps,
    depths,
    times,
    dd=None,
    tt=None,
    sigma=None,
    destripe=False,
    return_N=False,
):
    # could do [0, D] instead of [min depth, max depth]
    # dd = np.linspace(depths.min(), depths.max(), num=D)
    if dd is None:
        D = 1 + int(np.floor(depths.max()))
        dd = np.arange(0, D)
    else:
        D = dd.size

    if tt is None:
        T = 1 + int(np.floor(times.max()))
        tt = np.arange(0, T)

    M = np.zeros((D, T))
    N = np.zeros((D, T))
    R = np.zeros((D, T))
    times = np.ascontiguousarray(times, dtype=np.float64)
    depths = np.ascontiguousarray(depths, dtype=np.float64)
    amps = np.ascontiguousarray(amps, dtype=np.float64)
    _raster(times, depths, amps, M, N)

    Nnz = np.nonzero(N)
    R[Nnz] = M[Nnz] / N[Nnz]

    if destripe:
        R = wtdestripe(R)

    if sigma is not None:
        nz = np.flatnonzero(R.max(axis=1))
        if nz.size > 0:
            R[nz, :] = cheap_anscombe_denoising(
                R[nz, :], estimate_sig=False, sigma=sigma
            )

    if return_N:
        return R, N, dd, tt

    return R, dd, tt


def wtdestripe(raster):
    D, W = raster.shape
    LL0 = raster
    wlet = "db5"
    coeffs = pywt.wavedec2(LL0, wlet)
    L = len(coeffs)
    for i in range(1, L):
        HL = coeffs[i][1]
        Fb = np.fft.fft2(HL)
        Fb = np.fft.fftshift(Fb)
        mid = Fb.shape[0] // 2
        Fb[mid, :] = 0
        Fb[mid - 1, :] /= 3
        Fb[mid + 1, :] /= 3
        Fb = np.fft.ifftshift(Fb)
        coeffs[i] = (coeffs[i][0], np.real(np.fft.ifft2(Fb)), coeffs[i][2])
    LL = pywt.waverec2(coeffs, wlet)
    LL = np.ascontiguousarray(LL[:D, :W], dtype=raster.dtype)
    return LL


def cheap_anscombe_denoising(
    z,
    sigma=1,
    h=0.1,
    estimate_sig=True,
    fast_mode=True,
    multichannel=False,
    patch_size=5,
    patch_distance=5,
):
    minmax = (z - z.min()) / (z.max() - z.min())  # scales data to 0-1

    # Gaussianizing Poissonian data
    z_anscombe = 2.0 * np.sqrt(minmax + (3.0 / 8.0))

    if estimate_sig:
        sigma = np.mean(estimate_sigma(z_anscombe, multichannel=multichannel))
        print(f"estimated sigma: {sigma}")

    # Gaussian denoising
    z_anscombe_denoised = denoise_nl_means(
        z_anscombe,
        h=h * sigma,
        sigma=sigma,
        fast_mode=fast_mode,
        patch_size=patch_size,
        patch_distance=patch_distance,
    )

    z_inverse_anscombe = (
        (z_anscombe_denoised / 2.0) ** 2
        + 0.25 * np.sqrt(1.5) * z_anscombe_denoised ** -1
        - (11.0 / 8.0) * z_anscombe_denoised ** -2
        + (5.0 / 8.0) * np.sqrt(1.5) * z_anscombe_denoised ** -3
        - (1.0 / 8.0)
    )

    z_inverse_anscombe_scaled = (z.max() - z.min()) * (
        z_inverse_anscombe - z_inverse_anscombe.min()
    )

    return z_inverse_anscombe_scaled
