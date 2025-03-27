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
