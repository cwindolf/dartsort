import torch
import torch.nn.functional as F


def translate(input, shifts, mode='bilinear', padding_mode='zeros'):
    """Translate a batch of images by a batch of xy shifts

    Arguments
    ---------
    input : torch.Tensor NCHW
    shifts : torch.Tensor (N, 2)
        Shifts, in units of pixels (not these torch [-1, 1] coords)
    """
    N, C, H, W = input.shape
    N_, two = shifts.shape
    assert N == N_
    assert two == 2

    # homogeneous coordinate transforms. they should get a chiller api.
    theta = torch.eye(3).view(-1, 3, 3).tile((N, 1, 1))
    theta[:, :2, 2] = shifts

    # build and do transforms
    grid = F.affine_grid(theta, input.shape)
    output = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode)

    return output
