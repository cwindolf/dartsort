import numpy as np
from spikeinterface import BaseRecording
from fastplotlib.widgets.nd_widget import NDPositionsProcessor


class NDSpikeInterfaceProcessor(NDPositionsProcessor):
    def __init__(
            self,
            data: BaseRecording,
            dims: tuple[str, str, str],
            max_display_datapoints: int = 1_000,
            **kwargs,
    ):
        spatial_dims = dims,  # this is always going to be [l, p, 2] for spikeinterface

        super().__init__(
            data=data,
            max_display_datapoints=max_display_datapoints,
            **kwargs,
        )

    @property
    def data(self) -> BaseRecording:
        return self._data

    @property
    def multi(self) -> bool:
        return True

    @multi.setter
    def multi(self, *args):
        pass

    def _validate_data(self, data: BaseRecording, dims):
        if not isinstance(data, BaseRecording):
            raise TypeError

        return data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.get_num_channels(), self.data.get_num_samples(), 2

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def n_slider_dims(self) -> int:
        return 1

    def get(self, indices: dict[str, float | int]) -> np.ndarray:
        # assume no additional slider dims, only time slider dim
        dw_slice = self._get_dw_slice(indices)

        # slice xs
        xs = self.data.get_times()[dw_slice]

        start, stop = dw_slice.start, dw_slice.stop

        ys = self.data.get_traces(0, start, stop)

        return np.stack([np.broadcast_to(xs[:, None], (xs.shape[0], ys.shape[1])), ys]).T
