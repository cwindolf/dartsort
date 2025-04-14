import dataclasses
import dartsort


def test_cfg_consistency():
    """Ensure config.py and internal_config.py don't diverge."""
    cfg0 = dartsort.to_internal_config(dartsort.DeveloperConfig())
    cfg1 = dartsort.DARTsortInternalConfig()

    # can just do assert cfg0 == cfg1, but pytest gives a better
    # error message if you do...
    for field in dataclasses.fields(dartsort.DARTsortInternalConfig):
        assert getattr(cfg0, field.name) == getattr(cfg1, field.name)


def test_waveform_config():
    cfg0 = dartsort.WaveformConfig()
    cfg1 = dartsort.WaveformConfig.from_samples(42, 79)

    assert cfg0 == cfg1
    assert cfg0.trough_offset_samples() == 42
    assert cfg0.spike_length_samples() == 121
    assert cfg1.trough_offset_samples() == 42
    assert cfg1.spike_length_samples() == 121
