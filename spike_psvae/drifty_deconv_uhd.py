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
    spike_train, z_abs, bin_size_um, t_end=100,
    units_spread=None, min_spikes_bin=None, n_spikes_max_recent = 1000, fs=30000,
):
    """
    remove min_spikes_bin by default - it's worse to end up with a template that is mean of all spikes!!!
    units_spread is the spread of each registered clusters np.std(z_reg[spt[:, 1]==unit])*1.65 
    """
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
    medians_at_computation = np.zeros(unit_labels.max()+1)
    cur_superres_label = 0
    for u in unit_labels:
        in_u = np.flatnonzero(spike_train[:, 1] == u)

        # Get most recent spikes
        count_unit = np.logical_and(spike_train[:, 0]<t_end*fs, spike_train[:, 1]==u).sum()
        if count_unit>n_spikes_max_recent:
            in_u = np.flatnonzero(np.logical_and(spike_train[:, 0]<t_end*fs, 
                                                           spike_train[:, 1]==u))[-n_spikes_max_recent:]
        else:
            in_u = np.flatnonzero(spike_train[:, 1]==u)[:n_spikes_max_recent]

        # center the z positions in this unit using the median
        centered_z = z_abs[in_u].copy()
        medians_at_computation[u] = np.median(centered_z)
        centered_z -= medians_at_computation[u]

        # convert them to bin identities by adding half the bin size and
        # floor dividing by the bin size
        # this corresponds to bins like:
        #      ... | bin -1 | bin 0 | bin 1 | ...
        #   ... -3bin/2 , -bin/2, bin/2, 3bin/2, ...
        bin_ids = (centered_z + bin_size_um / 2) // bin_size_um
        occupied_bins, bin_counts = np.unique(bin_ids, return_counts=True)
        if units_spread is not None:
            # np.abs(bin_ids) <= (np.abs(centered_z)+ bin_size_um / 2)//bin_size_um <= (max_z_dist + bin_size_um / 2)//bin_size_um
            bin_counts = bin_counts[
                np.abs(occupied_bins)
                <= (units_spread[u] + bin_size_um / 2) // bin_size_um
            ]
            occupied_bins = occupied_bins[
                np.abs(occupied_bins)
                <= (units_spread[u] + bin_size_um / 2) // bin_size_um
            ]
        # IS THAT NEEDED - min_spikes_bin=6
        if min_spikes_bin is None:
            for bin_id in occupied_bins:
                superres_labels[in_u[bin_ids == bin_id]] = cur_superres_label
                superres_label_to_bin_id.append(bin_id)
                superres_label_to_orig_label.append(u)
                cur_superres_label += 1
        else:
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
        medians_at_computation
    )



def superres_denoised_templates(
    spike_train,
    z_abs,
    bin_size_um,
    geom,
    raw_binary_file,
    t_end=100,
    min_spikes_bin=None,
    units_spread=None,
    max_spikes_per_unit=200, #per superres unit 
    n_spikes_max_recent = 1000,
    denoise_templates=True,
    do_temporal_decrease=True,
    zero_radius_um=70, #reduce this value in uhd compared to NP1/NP2
    reducer=np.median,
    snr_threshold=5.0 * np.sqrt(100),
    spike_length_samples=121,
    trough_offset=42,
    do_tpca=True,
    tpca=None,
    tpca_rank=5,
    tpca_radius=75,
    tpca_n_wfs=50_000,
    fs=30000,
    pbar=True,
    seed=0,
    n_jobs=-1,
):

    (
        superres_labels,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
        medians_at_computation
    ) = superres_spike_train(
        spike_train,
        z_abs,
        bin_size_um,
        t_end,
        units_spread,
        min_spikes_bin,
        n_spikes_max_recent,
        fs,
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
        medians_at_computation
    )



def shift_superres_templates(
    superres_templates,
    superres_label_to_bin_id,
    superres_label_to_orig_label,
    bin_size_um,
    geom,
    disp_value,
    registered_medians,
    medians_at_computation,
    fill_value=0.0,
):

    """
    This version shifts by every (possible - if enough templates) mod 
    """
    pitch = get_pitch(geom)

    shifted_templates = superres_templates.copy()

    #shift every unit separately
    for unit in np.unique(superres_label_to_orig_label):
        # shift in bins, rounded towards 0
        bins_shift = np.round((disp_value + registered_medians[unit] - medians_at_computation[unit]))
        if bins_shift!=0:
            # How to do the shifting?
            # We break the shift into two pieces: the number of full pitches,
            # and the remaining bins after shifting by full pitches.
            n_pitches_shift = int(
                bins_shift / pitch
            )  # want to round towards 0, not //

            bins_shift_rem = bins_shift - pitch * n_pitches_shift

            # Now, first we do the pitch shifts
            shifted_templates_unit = pitch_shift_templates(
                n_pitches_shift, geom, superres_templates[superres_label_to_orig_label==unit], fill_value=fill_value
            )
            # Now, do the mod shift bins_shift_rem
            # IDEA: take the bottom bin and shift it above - 
            # If more than pitch/2 templates - OK, can shift 
            # Only special case np.abs(bins_shift_rem)<=pitch/2 and n_temp <=pitch/2 -> better not to shift (no information gain)

            n_temp = (superres_label_to_orig_label==unit).sum()
            if bins_shift_rem<0:
                if bins_shift_rem<-pitch/2 or n_temp>pitch/2:
                    idx_mod_shift = np.flatnonzero(np.isin(superres_label_to_bin_id[superres_label_to_orig_label==unit], superres_label_to_bin_id[superres_label_to_orig_label==unit].min()-np.arange(-bins_shift_rem)+pitch))
                    n_temp_shift = len(idx_mod_shift)
                    # if n_temp_shift:
                    shifted_templates_unit[-n_temp_shift:] = pitch_shift_templates(
                        -1, geom, shifted_templates_unit[idx_mod_shift], fill_value=fill_value
                    ) 

                    # The rest of the shift is handled by updating bin ids
                    # This part doesn't matter for the recovered spike train, since
                    # the template doesn't change, but it could matter for z tracking
                    superres_label_to_bin_id[superres_label_to_orig_label==unit] = np.roll(superres_label_to_bin_id[superres_label_to_orig_label==unit], -len(idx_mod_shift))
            elif bins_shift_rem>0:
                if bins_shift_rem>pitch/2 or n_temp>pitch/2:
                    idx_mod_shift = np.flatnonzero(np.isin(superres_label_to_bin_id[superres_label_to_orig_label==unit], superres_label_to_bin_id[superres_label_to_orig_label==unit].max()+np.arange(bins_shift_rem)-pitch))
                    n_temp_shift = len(idx_mod_shift)
                    # if n_temp_shift:
                    shifted_templates_unit[:n_temp_shift] = pitch_shift_templates(
                        1, geom, shifted_templates_unit[idx_mod_shift], fill_value=fill_value
                    ) #shift by 1 pitch as we taked templates that are at max()+shift-pitch
                    # update bottom templates - we remove <= "space" at the bottom than we add on top

                    # The rest of the shift is handled by updating bin ids
                    # This part doesn't matter for the recovered spike train, since
                    # the template doesn't change, but it could matter for z tracking

                    # !!! That's an approximation - maybe we'll need to change if we do z tracking, shoul;d be fine for now - is ok if we have bins that are "continuous" per unit
                    superres_label_to_bin_id[superres_label_to_orig_label==unit] = np.roll(superres_label_to_bin_id[superres_label_to_orig_label==unit], n_temp_shift)
            shifted_templates[superres_label_to_orig_label==unit]=shifted_templates_unit

    return shifted_templates, superres_label_to_bin_id


def shift_deconv(
    raw_bin,
    geom,
    p, #displacement
    bin_size_um,
    registered_medians,
    superres_templates,
    medians_at_computation,
    superres_label_to_bin_id,
    superres_label_to_orig_label,
    deconv_dir=None,
    pfs=30_000,
    t_start=0,
    t_end=None,
    n_jobs=1,
    trough_offset=42,
    spike_length_samples=121,
    max_upsample=1, #param for UHD - can increase if speed ok 
    refractory_period_frames=10,
    deconv_threshold=500, #Important param to validate
    su_chan_vis=3, #Important param to validate
):
    # discover the pitch of the probe
    # this is the unit at which the probe repeats itself.
    # so for NP1, it's not every row, but every 2 rows!
    pitch = get_pitch(geom)
    print(f"a {pitch=}")

    # integer probe-pitch shifts at each time bin
    p = p[t_start : t_end if t_end is not None else len(p)]
    bin_shifts = (p + bin_size_um / 2) // bin_size_um * bin_size_um
    unique_shifts, shift_ids_by_time = np.unique(
        bin_shifts, return_inverse=True
    )

    # for each shift, get shifted templates
    shifted_templates = np.array(
        [
            shift_superres_templates(
                superres_templates,
                superres_label_to_bin_id,
                superres_label_to_orig_label,
                bin_size_um,
                geom,
                shift,
                registered_medians,
                medians_at_computation)
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
            threshold=deconv_threshold,
            vis_su=su_chan_vis,
            conv_approx_rank=5,
            n_processors=n_jobs,
            multi_processing=n_jobs > 1,
            upsample=max_upsample,
            lambd=0,
            allowed_scale=0,
            template_index_to_unit_id=superres_label_to_orig_label,
            refractory_period_frames=refractory_period_frames,
        )
        my_batches = np.flatnonzero(bin_shifts == shift)
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
            bin_shifts[((st_up[:, 0] - trough_offset) // pfs).astype(int)] == unique_shifts[which_shiftix]
        ).all()
        # print(f"{bid=} {shift_good=} {tsorted=} {bigger=} {pitchy=}")
        assert shift_good
        assert tsorted
        assert bigger
        if not pitchy:
            raise ValueError(
                f"{bid=} Not pitchy {np.unique(bin_shifts[((st_up[:, 0] - trough_offset) // pfs).astype(int)])=} "
                f"{which_shiftix=} {unique_shifts[which_shiftix]=} {np.unique((st_up[:, 0] - trough_offset) // pfs)=} "
                f"{bin_shifts[np.unique((st_up[:, 0] - trough_offset) // pfs)]=}"
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

def superres_deconv_chunk(
    raw_bin,
    geom,
    z_abs,
    p,
    spike_train,
    registered_medians=None, #registered_median
    units_spread=None, #registered_spread
    bin_size_um=1,
    deconv_dir=None,
    pfs=30_000,
    t_start=0,
    t_end=None,
    n_jobs=1,
    trough_offset=42,
    spike_length_samples=121,
    max_upsample=1,
    refractory_period_frames=10,
    min_spikes_bin=None,
    max_spikes_per_unit=200,
    tpca=None,
    deconv_threshold=500, #Important param to validate
    su_chan_vis=3, #Important param to validate
):

    Path(deconv_dir).mkdir(exist_ok=True)

    (
        superres_templates,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
        medians_at_computation,
    ) = superres_denoised_templates(
        spike_train,
        z_abs,
        bin_size_um,
        geom,
        raw_bin,
        t_end,
        min_spikes_bin,
        units_spread,
        max_spikes_per_unit,
        n_spikes_max_recent=1000,
        denoise_templates=True,
        do_temporal_decrease=True,
        zero_radius_um=70,
        reducer=np.median,
        snr_threshold=5.0 * np.sqrt(100),
        spike_length_samples=spike_length_samples,
        trough_offset=trough_offset,
        do_tpca=True,
        tpca=tpca,
        tpca_rank=5,
        tpca_radius=75,
        tpca_n_wfs=50_000,
        fs=pfs,
        seed=0,
        n_jobs=n_jobs,
    )

    shifted_deconv_res = shift_deconv(
        raw_bin,
        geom,
        p,
        bin_size_um,
        registered_medians,
        superres_templates,
        medians_at_computation,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
        deconv_dir=deconv_dir,
        pfs=pfs,
        t_start=t_start,
        t_end=t_end,
        n_jobs=n_jobs,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        max_upsample=max_upsample,
        refractory_period_frames=refractory_period_frames,
        deconv_threshold=deconv_threshold,
        su_chan_vis=su_chan_vis,
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
