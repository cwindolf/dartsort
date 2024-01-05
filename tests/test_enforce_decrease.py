import numpy as np
import torch
from dartsort.transform.enforce_decrease import (EnforceDecrease,
                                                 make_parents_index)
from dartsort.util.waveform_util import make_channel_index
from test_util import dense_layout


def test_make_parents_index():
    h = dense_layout()
    geom = np.c_[h["x"], h["y"]]
    channel_index = make_channel_index(geom, 100)
    parents_index = make_parents_index(geom, channel_index)
    assert parents_index.shape[0] == channel_index.shape[0]

    enfdec = EnforceDecrease(torch.as_tensor(channel_index), torch.as_tensor(geom))
    assert enfdec is not None
