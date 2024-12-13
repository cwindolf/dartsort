from .matching import ObjectiveUpdateTemplateMatchingPeeler


class UniversalTemplatesMatchingPeeler(ObjectiveUpdateTemplateMatchingPeeler):
    """KS-style universal-templates-from-data detection

    This tries to rephrase their algorithm as faithfully as possible
    using dartsort tools, for comparison purposes with our algorithms.

    The idea is to estimate some (they use 6, it turns out) single-channel
    shapes via K means applied to single-channel waveforms. These
    are then expanded out into a full template library by spatial
    convs with various Gaussians. Then, throw them into the matcher.
    Since KS' matcher has scale_std --> infty, we can put a large
    scale prior variance to match the spirit of the thing.

    TODO maybe I should implement scale_prior->infty in our matcher?
    """

    def __init__(self):
        pass
