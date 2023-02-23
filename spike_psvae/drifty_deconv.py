import h5py
import numpy as np
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing
from . import deconvolve, snr_templates, spike_train_utils, reassignment
from .waveform_utils import get_pitch, pitch_shift_templates
from .extract_deconv import extract_deconv


def superres_spike_train(
    spike_train, z_abs, bin_size_um, min_spikes_bin=10, max_z_dist=None
):
    assert spike_train.shape == (*z_abs.shape, 2)
    assert bin_size_um > 0

    # bin the spikes to create a binned "superres spike train"
    # we'll use this spike train in an expanded label space to compute templates
    superres_labels = np.full_like(spike_train[:, 1], -1)
    # this will keep track of which superres template corresponds to which bin,
    # information which we will need later to determine how to shift the templates
    superres_label_to_bin_id = []
    superres_label_to_orig_label = []
    unit_labels = np.unique(spike_train[spike_train[:, 1] >= 0, 1])
    cur_superres_label = 0
    for u in unit_labels:
        in_u = np.flatnonzero(spike_train[:, 1] == u)

        # center the z positions in this unit using the median
        centered_z = z_abs[in_u].copy()
        centered_z -= np.median(centered_z)

        # convert them to bin identities by adding half the bin size and
        # floor dividing by the bin size
        # this corresponds to bins like:
        #      ... | bin -1 | bin 0 | bin 1 | ...
        #   ... -3bin/2 , -bin/2, bin/2, 3bin/2, ...
        bin_ids = (centered_z + bin_size_um / 2) // bin_size_um
        occupied_bins, bin_counts = np.unique(bin_ids, return_counts=True)
        if max_z_dist is not None:
            # np.abs(bin_ids) <= (np.abs(centered_z)+ bin_size_um / 2)//bin_size_um <= (max_z_dist + bin_size_um / 2)//bin_size_um
            bin_counts = bin_counts[
                np.abs(occupied_bins)
                <= (max_z_dist + bin_size_um / 2) // bin_size_um
            ]
            occupied_bins = occupied_bins[
                np.abs(occupied_bins)
                <= (max_z_dist + bin_size_um / 2) // bin_size_um
            ]
        if bin_counts.max() >= min_spikes_bin:
            for bin_id in occupied_bins[bin_counts >= min_spikes_bin]:
                superres_labels[in_u[bin_ids == bin_id]] = cur_superres_label
                superres_label_to_bin_id.append(bin_id)
                superres_label_to_orig_label.append(u)
                cur_superres_label += 1
        # what if no template was computed for u
        else:
            superres_labels[in_u] = cur_superres_label
            superres_label_to_bin_id.append(0)
            superres_label_to_orig_label.append(u)
            cur_superres_label += 1

    superres_label_to_bin_id = np.array(superres_label_to_bin_id)
    superres_label_to_orig_label = np.array(superres_label_to_orig_label)
    return (
        superres_labels,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
    )


def superres_denoised_templates(
    spike_train,
    z_abs,
    bin_size_um,
    geom,
    raw_binary_file,
    min_spikes_bin=10,
    max_z_dist=None,
    max_spikes_per_unit=500,
    denoise_templates=True,
    do_temporal_decrease=True,
    zero_radius_um=200,
    reducer=np.median,
    snr_threshold=5.0 * np.sqrt(100),
    spike_length_samples=121,
    trough_offset=42,
    do_tpca=True,
    tpca=None,
    tpca_rank=5,
    tpca_radius=75,
    tpca_n_wfs=50_000,
    pbar=True,
    seed=0,
    n_jobs=-1,
):
    (
        superres_labels,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
    ) = superres_spike_train(
        spike_train,
        z_abs,
        bin_size_um,
        min_spikes_bin=min_spikes_bin,
        max_z_dist=max_z_dist,
    )
    templates, extra = snr_templates.get_templates(
        np.c_[spike_train[:, 0], superres_labels],
        geom,
        raw_binary_file,
        max_spikes_per_unit=max_spikes_per_unit,
        do_temporal_decrease=do_temporal_decrease,
        zero_radius_um=zero_radius_um,
        reducer=reducer,
        snr_threshold=snr_threshold,
        spike_length_samples=spike_length_samples,
        trough_offset=trough_offset,
        do_tpca=do_tpca,
        tpca=tpca,
        tpca_rank=tpca_rank,
        tpca_radius=tpca_radius,
        tpca_n_wfs=tpca_n_wfs,
        pbar=pbar,
        seed=seed,
        n_jobs=n_jobs,
        raw_only=not denoise_templates,
    )
    return (
        templates,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
    )


def shift_superres_templates(
    superres_templates,
    superres_label_to_bin_id,
    shift_um,
    bin_size_um,
    geom,
    fill_value=0.0,
):
    pitch = get_pitch(geom)
    bins_per_pitch = pitch / bin_size_um
    if bins_per_pitch != int(bins_per_pitch):
        raise ValueError(
            f"The pitch of this probe is {pitch}, but the bin size "
            f"{bin_size_um} does not evenly divide it."
        )

    # shift in bins, rounded towards 0
    bins_shift = int(shift_um / bin_size_um)
    if bins_shift == 0:
        return superres_templates, superres_label_to_bin_id

    # How to do the shifting?
    # We break the shift into two pieces: the number of full pitches,
    # and the remaining bins after shifting by full pitches.
    n_pitches_shift = int(
        bins_shift / bins_per_pitch
    )  # want to round towards 0, not //
    bins_shift_rem = bins_shift - bins_per_pitch * n_pitches_shift

    # Now, first we do the pitch shifts
    shifted_templates = pitch_shift_templates(
        n_pitches_shift, geom, superres_templates, fill_value=fill_value
    )

    # The rest of the shift is handled by updating bin ids
    # This part doesn't matter for the recovered spike train, since
    # the template doesn't change, but it could matter for z tracking
    superres_label_to_bin_id = superres_label_to_bin_id - bins_shift_rem

    return shifted_templates, superres_label_to_bin_id


def rigid_int_shift_deconv(
    raw_bin,
    geom,
    p,
    deconv_dir=None,
    reference_displacement=0,
    pfs=30_000,
    spike_train=None,
    templates=None,
    template_index_to_unit_id=None,
    threshold=30.0,
    # need to implement
    t_start=0,
    t_end=None,
    n_jobs=1,
    trough_offset=42,
    spike_length_samples=121,
    max_upsample=8,
    refractory_period_frames=10,
):
    # discover the pitch of the probe
    # this is the unit at which the probe repeats itself.
    # so for NP1, it's not every row, but every 2 rows!
    pitch = get_pitch(geom)
    print(f"a {pitch=}")

    # integer probe-pitch shifts at each time bin
    p = p[t_start : t_end if t_end is not None else len(p)]
    pitch_shifts = (p - reference_displacement + pitch / 2) // pitch
    unique_shifts, shift_ids_by_time = np.unique(
        pitch_shifts, return_inverse=True
    )

    # original denoised templates
    if templates is None:
        (
            spike_train,
            order,
            templates,
            template_shifts,
        ) = spike_train_utils.clean_align_and_get_templates(
            spike_train,
            geom.shape[0],
            raw_bin,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
        )
        templates, _ = snr_templates.get_templates(
            spike_train,
            geom,
            raw_bin,
            templates.ptp(1).argmax(1),
            n_jobs=n_jobs,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
        )

    # for each shift, get shifted templates
    shifted_templates = np.array(
        [
            pitch_shift_templates(shift, geom, templates)
            for shift in unique_shifts
        ]
    )

    # run deconv on just the appropriate batches for each shift
    deconv_dir = Path(
        deconv_dir if deconv_dir is not None else tempfile.mkdtemp()
    )
    if n_jobs > 1:
        ctx = multiprocessing.get_context("spawn")
    shifted_templates_up = []
    shifted_sparse_temp_map = []
    sparse_temp_to_orig_map = []
    batch2shiftix = {}
    for shiftix, (shift, temps) in enumerate(
        zip(unique_shifts, tqdm(shifted_templates, desc="Shifts"))
    ):
        mp_object = deconvolve.MatchPursuitObjectiveUpsample(
            templates=temps,
            deconv_dir=deconv_dir,
            standardized_bin=raw_bin,
            t_start=t_start,
            t_end=t_end,
            n_sec_chunk=1,
            sampling_rate=pfs,
            max_iter=1000,
            threshold=threshold,
            vis_su=1.0,
            conv_approx_rank=5,
            n_processors=n_jobs,
            multi_processing=n_jobs > 1,
            upsample=max_upsample,
            lambd=0,
            allowed_scale=0,
            template_index_to_unit_id=template_index_to_unit_id,
            refractory_period_frames=refractory_period_frames,
        )
        my_batches = np.flatnonzero(pitch_shifts == shift)
        for bid in my_batches:
            batch2shiftix[bid] = shiftix
        my_fnames = [
            deconv_dir / f"seg_{bid:06d}_deconv.npz" for bid in my_batches
        ]
        if n_jobs <= 1:
            mp_object.run(my_batches, my_fnames)
        else:
            with ctx.Pool(
                n_jobs,
                initializer=mp_object.load_saved_state,
            ) as pool:
                for res in tqdm(
                    pool.imap_unordered(
                        mp_object._run_batch,
                        zip(my_batches, my_fnames),
                    ),
                    total=len(my_batches),
                    desc="Template matching",
                ):
                    pass

        (
            templates_up,
            deconv_id_sparse_temp_map,
            sparse_id_to_orig_id,
        ) = mp_object.get_sparse_upsampled_templates(return_orig_map=True)
        templates_up = templates_up.transpose(2, 0, 1)
        shifted_templates_up.append(templates_up)
        shifted_sparse_temp_map.append(deconv_id_sparse_temp_map)
        sparse_temp_to_orig_map.append(sparse_id_to_orig_id)

        print(
            f"{templates_up.shape=} {deconv_id_sparse_temp_map.shape=} {sparse_id_to_orig_id.shape=}"
        )

        assert len(templates_up) == len(sparse_id_to_orig_id)

    # collect all shifted and upsampled templates
    shifted_upsampled_start_ixs = np.array(
        [0] + list(np.cumsum([t.shape[0] for t in shifted_templates_up[:-1]]))
    )
    all_shifted_upsampled_temps = np.concatenate(shifted_templates_up, axis=0)
    shifted_upsampled_idx_to_shift_id = np.concatenate(
        [[i] * t.shape[0] for i, t in enumerate(shifted_templates_up)], axis=0
    )
    shifted_upsampled_idx_to_orig_id = (
        np.concatenate(sparse_temp_to_orig_map, axis=0)
        # + shifted_upsampled_start_ixs[shifted_upsampled_idx_to_shift_id]
    )
    assert shifted_upsampled_idx_to_shift_id.shape == shifted_upsampled_idx_to_orig_id.shape

    # gather deconv resultsdeconv_st = []
    deconv_spike_train_shifted_upsampled = []
    deconv_spike_train = []
    deconv_scalings = []
    deconv_dist_metrics = []
    print("gathering deconvolution results")
    for bid in range(mp_object.n_batches):
        which_shiftix = batch2shiftix[bid]

        fname_out = deconv_dir / f"seg_{bid:06d}_deconv.npz"
        with np.load(fname_out) as d:
            st = d["spike_train"]
            deconv_scalings.append(d["scalings"])
            deconv_dist_metrics.append(d["dist_metric"])

        st[:, 0] += trough_offset

        # usual spike train
        deconv_st = st.copy()
        deconv_st[:, 1] //= max_upsample
        deconv_spike_train.append(deconv_st)

        # upsampled + shifted spike train
        st_up = st.copy()
        st_up[:, 1] = shifted_sparse_temp_map[which_shiftix][st_up[:, 1]]
        st_up[:, 1] += shifted_upsampled_start_ixs[which_shiftix]
        deconv_spike_train_shifted_upsampled.append(st_up)

        shift_good = (
            shifted_upsampled_idx_to_shift_id[st_up[:, 1]] == which_shiftix
        ).all()
        tsorted = (np.diff(st_up[:, 0]) >= 0).all()
        bigger = (bid == 0) or (
            st_up[:, 0] >= deconv_spike_train_shifted_upsampled[-2][:, 0].max()
        ).all()
        pitchy = (
            pitch_shifts[((st_up[:, 0] - trough_offset) // pfs).astype(int)] == unique_shifts[which_shiftix]
        ).all()
        # print(f"{bid=} {shift_good=} {tsorted=} {bigger=} {pitchy=}")
        assert shift_good
        assert tsorted
        assert bigger
        if not pitchy:
            raise ValueError(
                f"{bid=} Not pitchy {np.unique(pitch_shifts[((st_up[:, 0] - trough_offset) // pfs).astype(int)])=} "
                f"{which_shiftix=} {unique_shifts[which_shiftix]=} {np.unique((st_up[:, 0] - trough_offset) // pfs)=} "
                f"{pitch_shifts[np.unique((st_up[:, 0] - trough_offset) // pfs)]=}"
            )

    deconv_spike_train = np.concatenate(deconv_spike_train, axis=0)
    deconv_spike_train_shifted_upsampled = np.concatenate(
        deconv_spike_train_shifted_upsampled, axis=0
    )
    deconv_scalings = np.concatenate(deconv_scalings, axis=0)
    deconv_dist_metrics = np.concatenate(deconv_dist_metrics, axis=0)

    print(
        f"Number of Spikes deconvolved: {deconv_spike_train_shifted_upsampled.shape[0]}"
    )

    return dict(
        deconv_spike_train=deconv_spike_train,
        deconv_spike_train_shifted_upsampled=deconv_spike_train_shifted_upsampled,
        deconv_scalings=deconv_scalings,
        shifted_templates=shifted_templates,
        all_shifted_upsampled_temps=all_shifted_upsampled_temps,
        shifted_upsampled_idx_to_orig_id=shifted_upsampled_idx_to_orig_id,
        shifted_upsampled_idx_to_shift_id=shifted_upsampled_idx_to_shift_id,
        deconv_dist_metrics=deconv_dist_metrics,
    )


def superres_deconv(
    raw_bin,
    geom,
    z_abs,
    p,
    spike_train,
    bin_size_um=None,
    deconv_dir=None,
    pfs=30_000,
    reference_displacement=0,
    threshold=30.0,
    max_z_dist=None,
    t_start=0,
    t_end=None,
    n_jobs=1,
    trough_offset=42,
    spike_length_samples=121,
    max_upsample=8,
    refractory_period_frames=10,
    denoise_templates=True,
):
    Path(deconv_dir).mkdir(exist_ok=True)

    (
        superres_templates,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
    ) = superres_denoised_templates(
        spike_train,
        z_abs,
        bin_size_um,
        geom,
        raw_bin,
        denoise_templates=denoise_templates,
        min_spikes_bin=10,
        max_z_dist=max_z_dist,
        max_spikes_per_unit=500,
        do_temporal_decrease=True,
        zero_radius_um=200,
        reducer=np.median,
        snr_threshold=5.0 * np.sqrt(100),
        spike_length_samples=spike_length_samples,
        trough_offset=trough_offset,
        do_tpca=True,
        tpca=None,
        tpca_rank=5,
        tpca_radius=75,
        tpca_n_wfs=50_000,
        n_jobs=n_jobs,
    )

    shifted_deconv_res = rigid_int_shift_deconv(
        raw_bin,
        geom,
        p,
        deconv_dir=None,
        reference_displacement=reference_displacement,
        pfs=pfs,
        spike_train=None,
        templates=superres_templates,
        t_start=t_start,
        t_end=t_end,
        n_jobs=n_jobs,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        max_upsample=max_upsample,
        template_index_to_unit_id=superres_label_to_orig_label,
        refractory_period_frames=refractory_period_frames,
        threshold=threshold,
    )

    # unpack results
    deconv_dist_metrics = shifted_deconv_res["deconv_dist_metrics"]
    superres_deconv_spike_train = shifted_deconv_res["deconv_spike_train"]
    superres_deconv_spike_train_shifted_upsampled = shifted_deconv_res[
        "deconv_spike_train_shifted_upsampled"
    ]
    deconv_scalings = shifted_deconv_res["deconv_scalings"]
    all_shifted_upsampled_temps = shifted_deconv_res[
        "all_shifted_upsampled_temps"
    ]
    shifted_upsampled_idx_to_superres_id = shifted_deconv_res[
        "shifted_upsampled_idx_to_orig_id"
    ]
    shifted_upsampled_idx_to_shift_id = shifted_deconv_res[
        "shifted_upsampled_idx_to_shift_id"
    ]

    # back to original label space
    deconv_spike_train = superres_deconv_spike_train.copy()
    deconv_spike_train[:, 1] = superres_label_to_orig_label[
        deconv_spike_train[:, 1]
    ]
    shifted_upsampled_idx_to_orig_id = superres_label_to_orig_label[
        shifted_upsampled_idx_to_superres_id
    ]
    shifted_upsampled_idx_to_superres_bin_id = superres_label_to_bin_id[
        shifted_upsampled_idx_to_superres_id
    ]

    # return everything the user could need
    return dict(
        deconv_spike_train=deconv_spike_train,
        superres_deconv_spike_train=superres_deconv_spike_train,
        superres_deconv_spike_train_shifted_upsampled=superres_deconv_spike_train_shifted_upsampled,
        deconv_scalings=deconv_scalings,
        superres_templates=superres_templates,
        superres_label_to_orig_label=superres_label_to_orig_label,
        superres_label_to_bin_id=superres_label_to_bin_id,
        all_shifted_upsampled_temps=all_shifted_upsampled_temps,
        shifted_upsampled_idx_to_superres_id=shifted_upsampled_idx_to_superres_id,
        shifted_upsampled_idx_to_superres_bin_id=shifted_upsampled_idx_to_superres_bin_id,
        shifted_upsampled_idx_to_orig_id=shifted_upsampled_idx_to_orig_id,
        shifted_upsampled_idx_to_shift_id=shifted_upsampled_idx_to_shift_id,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        bin_size_um=bin_size_um,
        raw_bin=raw_bin,
        deconv_dir=deconv_dir,
        deconv_dist_metrics=deconv_dist_metrics,
        shifted_superres_templates=shifted_deconv_res["shifted_templates"],
    )


def superres_propose_pairs(
    superres_templates,
    superres_label_to_orig_label,
    method="resid_dist",
    max_resid_dist=8,
    resid_dist_kwargs=dict(
        lambd=0.001,
        allowed_scale=0.1,
        thresh_mul=0.9,
        normalized=True,
        n_jobs=-1,
    ),
    max_distance=50,
    loc_radius=100,
    geom=None,
):
    n_superres_templates = superres_templates.shape[0]
    assert superres_label_to_orig_label.shape == (n_superres_templates,)

    # which pairs of superres templates are close enough?
    # list of superres template indices of length n_superres_templates
    superres_pairs = reassignment.propose_pairs(
        superres_templates,
        method=method,
        max_resid_dist=max_resid_dist,
        resid_dist_kwargs=resid_dist_kwargs,
        max_distance=max_distance,
        loc_radius=loc_radius,
        geom=geom,
    )

    # which pairs of units have superres pairs which are close enough?
    # list of unit indices of length n_units = np.unique(superres_label_to_orig_label).size
    unit_pairs = [
        np.unique(
            [
                superres_label_to_orig_label[pair_unit]
                for superres_ix in np.flatnonzero(
                    superres_label_to_orig_label == unit
                )
                for pair_unit in superres_pairs[superres_ix]
            ]
        )
        for unit in np.unique(superres_label_to_orig_label)
    ]

    return unit_pairs, superres_pairs


def extract_superres_shifted_deconv(
    superres_deconv_result,
    overwrite=True,
    pbar=True,
    nn_denoise=True,
    output_directory=None,
    extract_radius_um=100,
    n_sec_train_feats=40,
    # superres_propose_pairs args
    pairs_method="resid_dist",
    max_resid_dist=8,
    resid_dist_kwargs=dict(
        lambd=0.001,
        allowed_scale=0.1,
        thresh_mul=0.9,
        normalized=True,
        n_jobs=-1,
    ),
    max_distance=50,
    # what to save / do?
    save_residual=False,
    save_cleaned_waveforms=False,
    save_denoised_waveforms=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    tpca_rank=8,
    tpca_weighted=False,
    localize=True,
    loc_radius=100,
    save_outlier_scores=True,
    do_reassignment=True,
    do_reassignment_tpca=True,
    save_reassignment_residuals=False,
    reassignment_tpca_rank=5,
    reassignment_norm_p=np.inf,
    reassignment_tpca_spatial_radius=75,
    reassignment_tpca_n_wfs=50000,
    # usual suspects
    sampling_rate=30000,
    n_sec_chunk=1,
    device=None,
    geom=None,
    subtraction_h5=None,
    n_jobs=-1,
):
    """
    This is a wrapper that helps us deal with the bookkeeping for proposed
    pairs and reassignment with the shifting and the superres and the
    upsampling and all that...
    """
    # propose pairs of unit and superres labels FOR REASSIGNMENT ONLY? SUPER HEAVY IN MEMORY...
    if do_reassignment:
        unit_pairs, superres_pairs = superres_propose_pairs(
            superres_deconv_result["superres_templates"],
            superres_deconv_result["superres_label_to_orig_label"],
            method=pairs_method,
            max_resid_dist=max_resid_dist,
            resid_dist_kwargs=resid_dist_kwargs,
            max_distance=max_distance,
            loc_radius=loc_radius,
            geom=geom,
        )
    else:
        superres_pairs = None
        unit_pairs = None

    # infer what upsampled shifted superres units can be pairs
    shifted_upsampled_idx_to_shift_id = superres_deconv_result[
        "shifted_upsampled_idx_to_shift_id"
    ]
    shifted_upsampled_idx_to_superres_id = superres_deconv_result[
        "shifted_upsampled_idx_to_superres_id"
    ]
    shifted_upsampled_idx_to_orig_id = superres_deconv_result[
        "shifted_upsampled_idx_to_orig_id"
    ]
    print(f"{shifted_upsampled_idx_to_superres_id.shape=}")
    # print(",".join(map(str, shifted_upsampled_idx_to_superres_id)))

    zero_upsampled_temps = np.flatnonzero(
        np.all(
            superres_deconv_result["all_shifted_upsampled_temps"] == 0,
            axis=(1, 2),
        )
    )

    shifted_upsampled_pairs = []
    if do_reassignment:
        for shifted_upsampled_idx, (shift_id, superres_id) in enumerate(
            zip(
                shifted_upsampled_idx_to_shift_id,
                shifted_upsampled_idx_to_superres_id,
            )
        ):
            if shifted_upsampled_idx in zero_upsampled_temps:
                # these don't matter, since they won't get any spikes, but let's
                # special case.
                shifted_upsampled_pairs.append(
                    np.array([shifted_upsampled_idx])
                )
                continue

            superres_matches = superres_pairs[superres_id]
            shifted_upsampled_matches = np.flatnonzero(
                (shifted_upsampled_idx_to_shift_id == shift_id)
                & np.isin(
                    shifted_upsampled_idx_to_superres_id, superres_matches
                )
            )
            shifted_upsampled_matches = np.setdiff1d(
                shifted_upsampled_matches, zero_upsampled_temps
            )

            # print("-----")
            # print(f"{shifted_upsampled_idx=} {shift_id=} {superres_id=}")
            # print(f"{superres_pairs[superres_id]=}")
            # print(f"{shifted_upsampled_matches=}")
            # print(f"{shifted_upsampled_idx_to_shift_id[shifted_upsampled_matches]=}")
            # print(f"{shifted_upsampled_idx_to_superres_id[shifted_upsampled_matches]=}")

            shifted_upsampled_pairs.append(shifted_upsampled_matches)

        nps = np.array(list(map(len, shifted_upsampled_pairs)))
        print(f"Median n pairs: {nps.min()=}, {np.median(nps)=}, {nps.max()=}")

    if output_directory is None:
        output_directory = superres_deconv_result["deconv_dir"]

    ret = extract_deconv(
        superres_deconv_result["all_shifted_upsampled_temps"],
        superres_deconv_result[
            "superres_deconv_spike_train_shifted_upsampled"
        ],
        output_directory,
        superres_deconv_result["raw_bin"],
        scalings=superres_deconv_result["deconv_scalings"],
        geom=geom,
        extract_radius_um=extract_radius_um,
        subtraction_h5=subtraction_h5,
        save_residual=save_residual,
        save_cleaned_waveforms=save_cleaned_waveforms,
        save_denoised_waveforms=save_denoised_waveforms,
        save_cleaned_tpca_projs=save_cleaned_tpca_projs,
        save_denoised_tpca_projs=save_denoised_tpca_projs,
        tpca_rank=tpca_rank,
        tpca_weighted=tpca_weighted,
        save_outlier_scores=save_outlier_scores,
        do_reassignment=do_reassignment,
        save_reassignment_residuals=save_reassignment_residuals,
        do_reassignment_tpca=do_reassignment_tpca,
        reassignment_proposed_pairs_up=shifted_upsampled_pairs,
        reassignment_tpca_rank=reassignment_tpca_rank,
        reassignment_norm_p=reassignment_norm_p,
        reassignment_tpca_spatial_radius=reassignment_tpca_spatial_radius,
        reassignment_tpca_n_wfs=reassignment_tpca_n_wfs,
        localize=localize,
        loc_radius=loc_radius,
        n_sec_train_feats=n_sec_train_feats,
        n_jobs=n_jobs,
        n_sec_chunk=n_sec_chunk,
        sampling_rate=sampling_rate,
        device=device,
        trough_offset=superres_deconv_result["trough_offset"],
        overwrite=overwrite,
        pbar=pbar,
        nn_denoise=nn_denoise,
        seed=0,
    )
    if save_residual:
        extract_h5, residual = ret
    else:
        extract_h5 = ret

    with h5py.File(extract_h5, "r+") as h5:
        # map the reassigned spike train from "shifted superres" label space
        # to both superres and the original label space, and store for user
        if do_reassignment:
            new_labels_shifted_up = h5["reassigned_labels_up"][:]
            new_labels_superres = shifted_upsampled_idx_to_superres_id[
                new_labels_shifted_up
            ]
            new_labels_orig = shifted_upsampled_idx_to_orig_id[
                new_labels_shifted_up
            ]
            reassigned_pct = 100 * np.mean(
                new_labels_orig
                != superres_deconv_result["deconv_spike_train"][:, 1]
            )
            print(f"{reassigned_pct:0.1f}% of spikes were reassigned.")
            h5.create_dataset(
                "reassigned_superres_labels", data=new_labels_superres
            )
            h5.create_dataset("reassigned_unit_labels", data=new_labels_orig)
            h5.create_dataset(
                "reassigned_shifted_upsampled_labels",
                data=new_labels_shifted_up,
            )

        # store everything also for the user
        for key in (
            "deconv_spike_train",
            "superres_deconv_spike_train",
            "superres_deconv_spike_train_shifted_upsampled",
            "superres_templates",
            "superres_label_to_orig_label",
            "superres_label_to_bin_id",
            "all_shifted_upsampled_temps",
            "shifted_upsampled_idx_to_superres_id",
            "shifted_upsampled_idx_to_orig_id",
            "shifted_upsampled_idx_to_shift_id",
            "bin_size_um",
            "deconv_dist_metrics",
        ):
            h5.create_dataset(key, data=superres_deconv_result[key])

    extra = dict(
        unit_pairs=unit_pairs,
        superres_pairs=superres_pairs,
        shifted_upsampled_pairs=shifted_upsampled_pairs,
    )

    if save_residual:
        return extract_h5, residual, extra
    else:
        return extract_h5, extra
