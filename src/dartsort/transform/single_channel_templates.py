import torch

from dartsort.util import waveform_util
from dartsort.util.universal_util import get_singlechan_centroids

from .transform_base import BaseWaveformModule


class SingleChannelTemplates(BaseWaveformModule):
    """Utility class for computing single channel templates"""

    default_name = "singlechan_templates"

    def __init__(
        self,
        channel_index,
        geom=None,
        n_centroids=10,
        pca_rank=8,
        spike_length_samples=121,
        trough_offset_samples=42,
        dtype=torch.float,
        alignment_padding=20,
        random_seed=0,
        kmeanspp_initial="random",
        taper=True,
        name=None,
        name_prefix=None,
    ):
        super().__init__(
            channel_index=channel_index, name=name, name_prefix=name_prefix
        )

        # main control
        self.n_centroids = n_centroids
        self.alignment_padding = alignment_padding
        self.taper = taper
        self.pca_rank = pca_rank

        # input waveform details
        self.channel_index = channel_index
        self.spike_length_samples = spike_length_samples
        self.trough_offset_samples = trough_offset_samples

        # output details
        self.template_length = spike_length_samples - 2 * alignment_padding
        self.template_trough = trough_offset_samples - alignment_padding

        # gizmos
        self.random_seed = random_seed
        self.kmeanspp_initial = kmeanspp_initial
        self._needs_fit = True

    def needs_fit(self):
        return self._needs_fit

    def fit(self, waveforms, max_channels, recording=None):
        singlechan_waveforms = waveform_util.grab_main_channels_torch(
            waveforms, max_channels, self.channel_index
        )
        assert singlechan_waveforms.ndim == 3
        assert singlechan_waveforms.shape[2] == 1
        singlechan_waveforms = singlechan_waveforms[:, :, 0]
        templates = get_singlechan_centroids(
            singlechan_waveforms=singlechan_waveforms,
            trough_offset_samples=self.template_trough,
            spike_length_samples=self.template_length,
            alignment_padding=self.alignment_padding,
            n_centroids=self.n_centroids,
            pca_rank=self.pca_rank,
            taper=self.taper,
            taper_start=self.alignment_padding // 2,
            taper_end=self.alignment_padding,
            kmeanspp_initial=self.kmeanspp_initial,
            random_seed=self.random_seed,
        )
        self.register_buffer("templates", templates)
        self._needs_fit = False
