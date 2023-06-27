import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


class EnforceDecrease(torch.nn.Module):
    """A torch module for enforcing spatial decrease of amplitudes

    Calling an instance of this class on a N,T,C batch of waveforms
    will return a result whose peak-to-peak amplitudes decrease
    as you move away from the detection channel.
    """

    def __init__(self, geom, channel_index):
        super().__init__()
        self.register_buffer(
            "parents_index",
            torch.LongTensor(make_parents_index(geom, channel_index)),
        )
        self.register_buffer(
            "_1",
            torch.tensor(1.0),
        )

    def forward(self, waveforms, max_channels):
        """
        enfdec = EnforceDecrease(geom, channel_index)
        ...
        dec_wfs, dec_ptps = enfdec(waveforms, maxchans)
        """
        n = waveforms.shape[0]
        assert (n,) == max_channels.shape
        assert waveforms.shape[2] == self.parents_index.shape[1]

        # get peak to peak amplitudes -- (N, c) shaped
        ptps = waveforms.max(dim=1).values - waveforms.min(dim=1).values

        # pad with an extra channel to support indexing tricks
        pad_ptps = F.pad(ptps, (0, 1), value=torch.inf)
        # get amplitudes of all parents for all channels -- (N, c, <=c-1)
        # TODO it may be possible to refactor using the new torch.Tensor.scatter_reduce_!
        parent_ptps = pad_ptps[
            torch.arange(n)[:, None, None],
            self.parents_index[max_channels],
        ]
        parent_min_ptps = parent_ptps.min(dim=2).values

        # what would we need to multiply by to ensure my amp is <= all parents?
        rescaling = torch.minimum(parent_min_ptps / ptps, self._1)
        decreasing_ptps = ptps * rescaling
        decreasing_waveforms = waveforms * rescaling[:, None, :]
        return decreasing_waveforms, decreasing_ptps


def make_parents_index(geom, channel_index):
    """Indexing helper for EnforceDecrease

    Waveforms extracted according to channel_index enter into the enforce
    decrease routine, and for each channel A in the waveform we want to find
    the maximum amplitude of any channel which is A's ancestor in the tree
    rooted at the detection channel. This function computes an indexing
    structure to help with that.

    If channel_index is (C, c)-shaped, where C is the full number of channels
    and c is the number of channels extracted per waverform, then this function
    returns a (C, c, <=c-1)-shaped array parents_index, where parents_index[a, b]
    contains the (at most) c-1 parents (relatively indexed) of channel b
    (relatively indexed) in the tree rooted at detection channel a.

    For each site A, another site B is considered a parent relative to detection
    channel C if it is inside the square whose diagonal is AC, which is tested
    by <ACB < 45 deg and <CBA > 90 deg.

    Here, relatively indexed means in the 0:c index space rather than the 0:C
    index space.
    """
    C, c = channel_index.shape
    assert geom.ndim == 2 and geom.shape[0] == C

    parents_index = np.full((C, c, c - 1), c)
    geom_pad = np.pad(geom, [(0, 1), (0, 0)], constant_values=np.nan)
    for i in range(len(geom)):
        g = geom_pad[channel_index[i]]
        i_rel = np.flatnonzero(channel_index[i] == i)
        assert i_rel.size == 1
        i_rel = i_rel[0]

        # detection channel = C
        # candidate parent = B
        # leaf = A
        g_detect = geom[i][None]
        g_rel = g - g_detect

        # compute all angles ACB
        # print(i)
        cos_across_detect = 1.0 - cdist(g_rel, g_rel, metric="cosine")
        cos_across_detect[i_rel, :] = cos_across_detect[:, i_rel] = 1
        # print(np.array2string(cos_across_detect, max_line_width=200))

        # compute all angles CBA
        # these are segments BA (parent is on the second axis)
        # child,parent
        g_par_to_child = g[:, None] - g[None, :]
        # segments BC (parent on first+only axis)
        g_par_to_detect = g_detect - g
        cos_across_parent = np.sum(g_par_to_detect[None] * g_par_to_child, axis=2)
        cos_across_parent[:, i_rel] = -1
        # don't need to normalize this one since we just care about sign of cosine

        # check valid -- child on first axis, candidate parent on second
        is_parent = (cos_across_detect >= np.cos(np.pi / 4))
        is_parent &= cos_across_parent < 0
        for j in range(c):
            my_parents = np.setdiff1d(np.flatnonzero(is_parent[j]), [j])
            parents_index[i, j, : my_parents.size] = my_parents

    # trim to largest number of parents
    biggest = (parents_index < c).sum(axis=2).max()
    parents_index = parents_index[:, :, :biggest]

    return parents_index
