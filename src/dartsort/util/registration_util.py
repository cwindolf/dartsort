try:
    from dredge import dredge_ap
    have_dredge = True
except ImportError:
    have_dredge = False
    pass


def estimate_motion(recording, sorting, motion_estimation_config=None, localizations_dataset_name="point_source_localizations"):
    if not motion_estimation_config.do_motion_estimation:
        return None

    if not have_dredge:
        raise ValueError("Please install DREDge to use motion estimation.")
