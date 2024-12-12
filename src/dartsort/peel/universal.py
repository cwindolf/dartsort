"""KS-style universal-templates-from-data detection

This tries to rephrase their algorithm as faithfully as possible
using our tools, for comparison purposes with our own algorithms.

The idea is to estimate some (well, 6, it turns out) single-channel
shapes via K means applied to single-channel waveforms. These
are then expanded out into a full template library by spatial
convs with various Gaussians. Then, throw them into the matcher.
Since KS' matcher has scale_std --> infty, we can just put a large
scale prior to match the spirit of the thing.
"""
