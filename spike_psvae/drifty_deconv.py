import numpy as np
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing
from . import deconvolve, snr_templates, spike_train_utils
from .waveform_utils import get_pitch, pitch_shift_templates


def superres_spike_train(
    spike_train,
    z_abs,
    bin_size_um,
    min_spikes_bin=10,
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
        for bin_id in occupied_bins[bin_counts >= min_spikes_bin]:
            superres_labels[in_u[bin_ids == bin_id]] = cur_superres_label
            superres_label_to_bin_id.append(bin_id)
            superres_label_to_orig_label.append(u)
            cur_superres_label += 1

    superres_label_to_bin_id = np.array(superres_label_to_bin_id)
    superres_label_to_orig_label = np.array(superres_label_to_orig_label)
    return superres_labels, superres_label_to_bin_id, superres_label_to_orig_label


def superres_denoised_templates(
    spike_train,
    z_abs,
    bin_size_um,
    geom,
    raw_binary_file,
    min_spikes_bin=10,
    max_spikes_per_unit=500,
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
    superres_labels, superres_label_to_bin_id, superres_label_to_orig_label = superres_spike_train(
        spike_train,
        z_abs,
        bin_size_um,
        min_spikes_bin=min_spikes_bin,
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
    )
    return templates, superres_label_to_bin_id, superres_label_to_orig_label


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
    n_pitches_shift = int(bins_shift / bins_per_pitch)  # want to round towards 0, not //
    bins_shift_rem = bins_shift - bins_per_pitch * n_pitches_shift

    # Now, first we do the pitch shifts
    shifted_templates = pitch_shift_templates(n_pitches_shift, geom, superres_templates, fill_value=fill_value)

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
    # need to implement
    # t_start=0,
    # t_end=None,
    n_jobs=1,
    trough_offset=42,
    spike_length_samples=121,
    max_upsample=8,
):
    # discover the pitch of the probe
    # this is the unit at which the probe repeats itself.
    # so for NP1, it's not every row, but every 2 rows!
    pitch = get_pitch(geom)
    print(f"{pitch=}")

    # integer probe-pitch shifts at each time bin
    pitch_shifts = (p - reference_displacement + pitch / 2) // pitch
    unique_shifts = np.unique(pitch_shifts)

    # original denoised templates
    if templates is None:
        spike_train, order, templates, template_shifts = spike_train_utils.clean_align_and_get_templates(
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
    shifted_templates = [
        pitch_shift_templates(shift, geom, templates) for shift in unique_shifts
    ]

    # run deconv on just the appropriate batches for each shift
    deconv_dir = Path(deconv_dir if deconv_dir is not None else tempfile.mkdtemp())
    if n_jobs > 1:
        ctx = multiprocessing.get_context("spawn")
    shifted_templates_up = []
    shifted_sparse_temp_map = []
    batch2shiftix = {}
    for shiftix, (shift, temps) in enumerate(zip(unique_shifts, tqdm(shifted_templates, desc="Shifts"))):
        mp_object = deconvolve.MatchPursuitObjectiveUpsample(
            templates=temps,
            deconv_dir=deconv_dir,
            standardized_bin=raw_bin,
            t_start=0,
            t_end=None,
            n_sec_chunk=1,
            sampling_rate=pfs,
            max_iter=1000,
            threshold=50,
            vis_su=1.0,
            conv_approx_rank=5,
            n_processors=n_jobs,
            multi_processing=n_jobs > 1,
            upsample=max_upsample,
            lambd=0,
            allowed_scale=0,
        )
        my_batches = np.flatnonzero(pitch_shifts == shift)
        for bid in my_batches:
            batch2shiftix[bid] = shiftix
        my_fnames = [deconv_dir / f"seg_{bid:06d}_deconv.npz" for bid in my_batches]
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
        ) = mp_object.get_sparse_upsampled_templates()
        shifted_templates_up.append(templates_up.transpose(2, 0, 1))
        shifted_sparse_temp_map.append(deconv_id_sparse_temp_map)

    # collect all shifted and upsampled templates
    shift_up_start_ixs = [0] + list(np.cumsum([t.shape[0] for t in shifted_templates_up]))
    all_up_temps = np.concatenate(shifted_templates_up, axis=0)

    # gather deconv resultsdeconv_st = []
    deconv_spike_train_up = []
    deconv_spike_train = []
    deconv_scalings = []
    print("gathering deconvolution results")
    for bid in range(mp_object.n_batches):
        which_shiftix = batch2shiftix[bid]

        fname_out = deconv_dir / f"seg_{bid:06d}_deconv.npz"
        with np.load(fname_out) as d:
            st = d["spike_train"]
            deconv_scalings.append(d["scalings"])

        st[:, 0] += trough_offset

        # usual spike train
        deconv_st = st.copy()
        deconv_st[:, 1] //= max_upsample
        deconv_spike_train.append(deconv_st)

        # upsampled + shifted spike train
        st_up = st.copy()
        st_up[:, 1] = shifted_sparse_temp_map[which_shiftix][st_up[:, 1]]
        st_up[:, 1] += shift_up_start_ixs[which_shiftix]
        deconv_spike_train_up.append(st_up)

    deconv_spike_train = np.concatenate(deconv_spike_train, axis=0)
    deconv_spike_train_up = np.concatenate(deconv_spike_train_up, axis=0)
    deconv_scalings = np.concatenate(deconv_scalings, axis=0)

    print(f"Number of Spikes deconvolved: {deconv_spike_train_up.shape[0]}")

    return deconv_spike_train, deconv_spike_train_up, deconv_scalings, all_up_temps


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
    # need to implement
    # t_start=0,
    # t_end=None,
    n_jobs=1,
    trough_offset=42,
    spike_length_samples=121,
    max_upsample=8,
):
    superres_templates, superres_label_to_bin_id, superres_label_to_orig_label = superres_denoised_templates(
        spike_train,
        z_abs,
        bin_size_um,
        geom,
        raw_bin,
        min_spikes_bin=10,
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
    print(superres_label_to_orig_label.size, superres_label_to_orig_label.max() + 1, superres_templates.shape)

    superres_deconv_spike_train, deconv_spike_train_up, deconv_scalings, all_up_temps = rigid_int_shift_deconv(
        raw_bin,
        geom,
        p,
        deconv_dir=None,
        reference_displacement=reference_displacement,
        pfs=pfs,
        spike_train=None,
        templates=superres_templates,
        # need to implement
        # t_start=0,
        # t_end=None,
        n_jobs=n_jobs,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        max_upsample=max_upsample,
    )

    # back to original label space
    deconv_spike_train = superres_deconv_spike_train.copy()
    deconv_spike_train[:, 1] = superres_label_to_orig_label[deconv_spike_train[:, 1]]

    return dict(
        deconv_spike_train=deconv_spike_train,
        superres_deconv_spike_train=superres_deconv_spike_train,
        superres_templates=superres_templates,
        superres_label_to_orig_label=superres_label_to_orig_label,
        deconv_spike_train_up=deconv_spike_train_up,
        all_up_temps=all_up_temps,
    )
