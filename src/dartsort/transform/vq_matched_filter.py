from typing import Literal

import numpy as np
import torch
from torch import Tensor

from dartsort.util.spiketorch import spawn_torch_rg

from ..clustering.kmeans import batched_kmeans
from ..util.logging_util import get_logger
from ..util.py_util import panic
from .temporal_pca import BaseTemporalPCA
from .transform_base import BaseWaveformModule

logger = get_logger(__name__)

FilterKind = Literal["centroid", "median", "rank1"]


class VQMatchedFilter(BaseTemporalPCA):
    default_name = "vq_matched_filter"

    def __init__(
        self,
        n_filters: int = 4,
        filter_kind: FilterKind = "centroid",
        weight_power: float = 2.0,
        max_snippets: int = 200_000,
        n_kmeans_tries: int = 5,
        n_kmeans_iter: int = 100,
        **super_kwargs,
    ):
        if filter_kind not in ("centroid", "median", "rank1"):
            panic(f"unknown {filter_kind=}")
        super().__init__(rank=n_filters, centered=False, whiten=False, **super_kwargs)
        self.n_filters = n_filters
        self.filter_kind = filter_kind
        self.weight_power = weight_power
        self.max_snippets = max_snippets
        self.n_kmeans_tries = n_kmeans_tries
        self.n_kmeans_iter = n_kmeans_iter
        self.cluster_sizes: list[int] = []

    @property
    def filters(self) -> Tensor:
        return self.b.components

    def get_extra_state(self):
        es = super().get_extra_state()
        es["cluster_sizes"] = list(self.cluster_sizes)
        return es

    def set_extra_state(self, state):
        super().set_extra_state(state)
        self.cluster_sizes = list(state.get("cluster_sizes", []))

    def fit(
        self,
        recording,
        waveforms,
        *,
        computation_cfg,
        channels,
        **spike_data,
    ):
        BaseWaveformModule.fit(
            self,
            recording,
            waveforms,
            computation_cfg=computation_cfg,
            channels=channels,
        )

        snippets = self.extract_snippets(
            waveforms,
            channels,
            weights=spike_data.get("weights"),
            time_shifts=spike_data.get("time_shifts"),
        )
        filters, sizes = self.fit_filters(snippets)
        self.set_filters(filters, sizes)

    def set_filters(self, filters: Tensor, sizes) -> None:
        if filters.shape[0] != self.b.components.shape[0]:
            self.b.components.resize_(filters.shape)
        self.b.components.copy_(filters)
        self.b.mean.zero_()
        self.b.whitener.resize_((filters.shape[0],))
        self.b.whitener.copy_(
            torch.as_tensor(sizes, dtype=self.b.whitener.dtype)
            / max(1, int(np.sum(sizes)))
        )
        self.rank = filters.shape[0]
        self.shape = (self.rank, self.b.channel_index.shape[1])
        self.cluster_sizes = [int(s) for s in sizes]
        self._needs_fit = False
        logger.dartsortdebug(
            f"Fit {self.rank} {self.filter_kind} filters of length "
            f"{filters.shape[1]}, cluster sizes {min(self.cluster_sizes)}"
            f"-{max(self.cluster_sizes)}."
        )

    def fit_filters(self, snippets: Tensor) -> tuple[Tensor, np.ndarray]:
        """(n, filter_length) snippets -> (k, filter_length)"""
        device = snippets.device
        norms = snippets.norm(dim=1)
        gen = spawn_torch_rg(device=device, seed=self.random_seed)

        n_sample = min(len(snippets), self.max_snippets)
        if self.weight_power:
            weights = norms.pow(self.weight_power)
            ix = torch.multinomial(weights, n_sample, replacement=True, generator=gen)
        else:
            ix = torch.randint(len(snippets), (n_sample,), device=device, generator=gen)
        x = snippets[ix]
        x = norms[ix].median() * x / norms[ix][:, None].clamp_min(1e-8)

        if self.n_filters == 1:
            labels = torch.zeros(len(x), dtype=torch.long, device=device)
            centroids = x.mean(0, keepdim=True)
        else:
            res = batched_kmeans(
                x,
                n_components=self.n_filters,
                seed=self.random_seed,
                n_iter=self.n_kmeans_iter,
                n_tries=self.n_kmeans_tries,
            )
            assert res.labels is not None
            assert res.centroids is not None
            labels = res.labels
            centroids = res.centroids

        bank = []
        sizes = []
        for j in range(len(centroids)):
            in_cluster = labels == j
            n_j = int(in_cluster.sum())
            if not n_j:
                continue
            sizes.append(n_j)
            bank.append(self._cluster_filter(x[in_cluster], centroids[j]))
        return torch.stack(bank), np.array(sizes)

    def _cluster_filter(self, x_j: Tensor, centroid: Tensor) -> Tensor:
        if self.filter_kind == "centroid":
            f = centroid
        elif self.filter_kind == "median":
            f = x_j.median(0).values
        elif self.filter_kind == "rank1":
            _, vecs = torch.linalg.eigh(x_j.T @ x_j)
            f = vecs[:, -1]
        else:
            panic(f"unknown {self.filter_kind=}")
        # sign ambiguity
        f = f * -torch.sign(f[f.abs().argmax()])
        return f / f.norm()
