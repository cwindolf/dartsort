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
        trough_offset_samples=42,
        dtype=torch.float,
        alignment_padding=20,
        random_state=0,
        kmeanspp_initial="random",
        taper=True,
        name=None,
        name_prefix=None,
        max_waveforms=None,
    ):
        super().__init__(
            channel_index=channel_index, name=name, name_prefix=name_prefix
        )

        # main control
        self.n_centroids = n_centroids
        self.alignment_padding = alignment_padding
        self.taper = taper
        self.pca_rank = pca_rank
        self.max_waveforms = max_waveforms

        # input waveform details
        self.channel_index = channel_index
        self.trough_offset_samples = trough_offset_samples
        self.alignment_padding = alignment_padding

        # gizmos
        self.random_state = random_state
        self.kmeanspp_initial = kmeanspp_initial
        self._needs_fit = True

    def needs_fit(self):
        return self._needs_fit

    def initialize_spike_length_dependent_params(self):
        self.template_trough = self.trough_offset_samples - self.alignment_padding
        self.template_length = self.spike_length_samples - 2 * self.alignment_padding
        self.register_buffer(
            "templates", torch.zeros((self.n_centroids, self.template_length))
        )
        self.to(self.channel_index.device)

    def fit(self, waveforms, max_channels, recording=None, weights=None):
        super().fit(waveforms, max_channels, recording, weights)
        if weights is not None:
            rg = np.random.default_rng(self.random_state)
            weights = weights.numpy(force=True) if torch.is_tensor(weights) else weights
            weights = weights.astype(np.float64)
            weights = weights / weights.sum()
            choices = rg.choice(len(weights), p=weights, size=self.max_waveforms)
            choices.sort()
            choices = torch.from_numpy(choices)
            waveforms = waveforms[choices]
            max_channels = max_channels[choices]
        singlechan_waveforms = waveform_util.grab_main_channels_torch(
            waveforms, max_channels, self.channel_index
        )
        assert singlechan_waveforms.ndim == 3
        assert singlechan_waveforms.shape[2] == 1
        singlechan_waveforms = singlechan_waveforms[:, :, 0]
        print(f"{waveforms.requires_grad=}")
        print(f"{singlechan_waveforms.requires_grad=}")
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
            random_seed=self.random_state,
        )
        self.templates[:] = templates
        self._needs_fit = False
