import torch
from torch import nn


class SingleChanDenoiser(nn.Module):
    """Cleaned up a little. Why is conv3 here and commented out in forward?"""

    def __init__(
        self, n_filters=[16, 8, 4], filter_sizes=[5, 11, 21], spike_size=121
    ):
        super(SingleChanDenoiser, self).__init__()
        feat1, feat2, feat3 = n_filters
        size1, size2, size3 = filter_sizes
        self.conv1 = nn.Sequential(nn.Conv1d(1, feat1, size1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(feat1, feat2, size2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(feat2, feat3, size3), nn.ReLU())
        n_input_feat = feat2 * (spike_size - size1 - size2 + 2)
        self.out = nn.Linear(n_input_feat, spike_size)

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)

    def load(self, fname_model="../pretrained/single_chan_denoiser.pt"):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint)
