import torch
from torch import Tensor


@torch.jit.script
def subtract_precomputed_pconv(
    conv: Tensor,
    pconv: Tensor,
    template_indices: Tensor,
    upsampling_indices: Tensor,
    scalings: Tensor,
    times: Tensor,
    padded_conv_lags: Tensor,
    neg: bool,
    batch_size: int = 128,
):
    ix_time = times[:, None] + padded_conv_lags[None, :]
    for i0 in range(0, conv.shape[0], batch_size):
        i1 = min(conv.shape[0], i0 + batch_size)
        batch = pconv[i0:i1, template_indices, upsampling_indices]
        batch.mul_(scalings[None, :, None])
        ix = ix_time.broadcast_to(batch.shape)
        batch = batch.reshape(i1 - i0, -1)
        ix = ix.reshape(i1 - i0, batch.shape[1])
        if neg:
            batch = batch._neg_view()
        conv[i0:i1].scatter_add_(dim=1, src=batch, index=ix)
