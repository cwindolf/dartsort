import argparse
import pickle
import shutil
import subprocess
import warnings
from pathlib import Path

import h5py
import numpy as np
import one.alf.io as alfio
import spikeinterface.core as sc
import spikeinterface.extractors as se
from brainbox.io.one import SpikeSortingLoader
from dredge import dredge_ap
from dredge import motion_util as mu
from one.api import ONE
from spike_psvae import subtract
from spike_psvae.grab_and_localize import grab_and_localize
from spike_psvae.spike_train_utils import clean_align_and_get_templates

warnings.simplefilter("ignore", category=DeprecationWarning)


sdsc_base_path = Path("/mnt/sdceph/users/ibl/data")


def save_motion_est(path, name, me):
    with open(path / f"{name}_motion_est.pkl", "wb") as jar:
        pickle.dump(me, jar)


def eid2sdscpath(eid):
    pids, probes = one.eid2pid(eid)
    print(pids, probes)
    alyx_base_path = one.eid2path(eid)
    print(alyx_base_path)
    paths = {}
    for pid, probe in zip(pids, probes):
        rel_path = one.list_datasets(eid, f"raw_ephys_data/{probe}*ap.cbin")
        print("rel_path", rel_path)
        assert len(rel_path) == 1
        rel_path = Path(rel_path[0])
        searchdir = (
            sdsc_base_path
            / alyx_base_path.relative_to(one.cache_dir)
            / rel_path.parent
        )
        pattern = Path(rel_path.name).with_suffix(f".*.cbin")
        glob = list(searchdir.glob(str(pattern)))
        assert len(glob) == 1
        paths[probe] = pid, glob[0]
        assert paths[probe][1].exists()
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("pids", type=str)
    ap.add_argument("--outdir", type=Path)
    ap.add_argument("--njobs", type=int, default=20)
    ap.add_argument("--nspca", type=int, default=40)
    ap.add_argument("--batchlen", type=float, default=1)
    ap.add_argument("--locworkers", type=int, default=2)
    ap.add_argument("--ksreloc", action="store_true")
    ap.add_argument("--ksmotion", action="store_true")
    ap.add_argument("--residnorm-decrease", type=float, default=np.sqrt(10.0))
    ap.add_argument("--tmp-parent", type=str, default="/tmp")
    ap.add_argument("--no-overwrite", action="store_true")
    ap.add_argument("--rerun-errors", action="store_true")

    args = ap.parse_args()

    one = ONE()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True, parents=True)
    subcache = Path(args.tmp_parent) / "subcache"
    dscache = Path(args.tmp_parent) / "dscache"
    dscache.mkdir(exist_ok=True)

    if Path(args.pids).exists():
        print("Found a PID file")
        args.pids = open(args.pids).read().split()
    else:
        args.pids = args.pids.strip().split(",")
    print("\n".join(args.pids))

    for pid in args.pids:
        eid, probe = one.pid2eid(pid)
        print("eid", eid, "probe", probe)
        pid_, cbin_path = eid2sdscpath(eid)[probe]
        assert pid_ == pid

        ch_path = list(cbin_path.parent.glob("*ap*.ch"))
        assert len(ch_path) == 1
        ch_path = ch_path[0]
        meta_path = list(cbin_path.parent.glob("*ap*.meta"))
        assert len(meta_path) == 1
        meta_path = meta_path[0]

        sessdir = outdir / f"pid{pid}"
        sessdir.mkdir(exist_ok=True)

        metadata = dict(
            probe=probe,
            eid=eid,
            pid=pid,
            cbin_path=cbin_path,
        )

        if (sessdir / "metadata.pkl").exists():
            with open(sessdir / "metadata.pkl", "rb") as sess_jar:
                meta = pickle.load(sess_jar)
                if "done" in meta and meta["done"]:
                    print(pid, "already done.")
                    if not (sessdir / "subtraction.h5").exists():
                        raise ValueError(
                            "Inconsistency with the done and no h5 for\n",
                            "meta:",
                            sessdir / "metadata.pkl",
                            "\n",
                            metadata,
                        )
                    # continue
                if "subtraction_error" in meta and not args.rerun_errors:
                    print(
                        "This one had a problem during subtraction in a previous run. Skipping"
                    )
                    continue
                if "subtraction_error" in meta and args.rerun_errors:
                    print(
                        "This one had a problem during subtraction in a previous run. Rerunning."
                    )

        with open(sessdir / "metadata.pkl", "wb") as sess_jar:
            pickle.dump(metadata, sess_jar)

        ch_path = list(cbin_path.parent.glob("*ap*.ch"))
        assert len(ch_path) == 1
        ch_path = ch_path[0]
        meta_path = list(cbin_path.parent.glob("*ap*.meta"))
        assert len(meta_path) == 1
        meta_path = meta_path[0]

        print("-" * 50)
        print(pid, cbin_path)
        rec_cbin = se.read_cbin_ibl(
            str(cbin_path.parent),
            cbin_file=str(cbin_path),
            ch_file=str(ch_path),
            meta_file=str(meta_path),
        )
        print(rec_cbin)
        fs = int(rec_cbin.get_sampling_frequency())

        do_anything = (not (sessdir / "ks_relocalizations.npz").exists()) or (
            not (sessdir / "subtraction.h5").exists()
        )
        if not do_anything:
            print("This one is done.")
            continue

        # copy to temp dir
        dscache.mkdir(exist_ok=True)
        cbin_rel = cbin_path.stem.split(".")[0] + (
            "".join(s for s in cbin_path.suffixes if len(s) < 10)
        )
        meta_rel = meta_path.stem.split(".")[0] + (
            "".join(s for s in meta_path.suffixes if len(s) < 10)
        )
        ch_rel = ch_path.stem.split(".")[0] + (
            "".join(s for s in ch_path.suffixes if len(s) < 10)
        )

        # run destriping
        destriped_bin = dscache / f"destriped_{cbin_rel}"
        destriped_bin = destriped_bin.with_suffix(".bin")
        if not destriped_bin.exists():
            shutil.copyfile(cbin_path, dscache / cbin_rel)
            shutil.copyfile(meta_path, dscache / meta_rel)
            shutil.copyfile(ch_path, dscache / ch_rel)

            try:
                subprocess.run(
                    [
                        "python",
                        str(Path(__file__).parent / "destripe.py"),
                        str(dscache / cbin_rel),
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                with open(sessdir / "metadata.pkl", "wb") as sess_jar:
                    metadata["destriping_error"] = e
                    pickle.dump(metadata, sess_jar)
                if (dscache / f"destriped_{cbin_rel}").exists():
                    (dscache / f"destriped_{cbin_rel}").unlink()
                continue
            finally:
                pass
                # for pfile in (cbin_rel, meta_rel, ch_rel):
                #     if (dscache / pfile).exists():
                #         (dscache / pfile).unlink()

        assert destriped_bin.exists()

        rec = sc.read_binary(
            destriped_bin,
            rec_cbin.get_sampling_frequency(),
            rec_cbin.get_num_channels(),
            dtype="float32",
        )
        rec.set_probe(rec_cbin.get_probe(), in_place=True)
        fs = rec.get_sampling_frequency()

        ttt = rec.get_traces(
            start_frame=rec.get_num_samples() // 2,
            end_frame=rec.get_num_samples() // 2 + 1000,
        )
        print(f"{ttt.min()=} {ttt.max()=}")

        # if subcache.exists():
        #     shutil.rmtree(subcache)

        if not (sessdir / "subtraction.h5").exists():
            print("Subtracting")
            try:
                sub_h5 = subtract.subtraction(
                    rec,
                    out_folder=subcache,
                    thresholds=[12, 10, 8, 6, 5],
                    n_sec_pca=args.nspca,
                    save_subtracted_tpca_projs=False,
                    save_cleaned_tpca_projs=False,
                    save_denoised_tpca_projs=False,
                    n_jobs=args.njobs,
                    loc_workers=args.locworkers,
                    overwrite=False,
                    n_sec_chunk=args.batchlen,
                    save_cleaned_pca_projs_on_n_channels=5,
                    loc_feature=("ptpgpu"),
                    residnorm_decrease=args.residnorm_decrease,
                )
                with h5py.File(sub_h5, "r+") as h5:
                    h5.create_dataset(
                        "localizations", data=h5["localizationsptpgpu"][:]
                    )
                    spike_times = h5["spike_index"][:, 0] + (
                        h5["start_time"][()] * 30_000
                    )
                    x = h5["localizations"][:, 0]
                    z = h5["localizations"][:, 2]
                    maxptp = h5["maxptps"][:]
                    geom = h5["geom"][:]
                    t = spike_times / 30000

                    wh = (
                        (z > geom[:, 1].min() - 250)
                        & (z < geom[:, 1].max() + 250)
                        & (x > geom[:, 0].min() - 250)
                        & (x < geom[:, 0].max() + 250)
                    )

                    tme, _ = dredge_ap.register(
                        maxptp[wh],
                        z[wh],
                        t[wh],
                        max_disp_um=50,
                        pbar=False,
                    )
                    save_motion_est(sessdir, "dredge_ap", tme)
                    z_reg = tme.correct_s(spike_times / 30000, z)
                    h5.create_dataset("z_reg", data=z_reg)

                    # ks' drift correction
                    if args.ksmotion:
                        ssl = SpikeSortingLoader(one=one, pid=pid)
                        ssl.download_spike_sorting_object("drift")
                        drift = alfio.load_object(
                            ssl.files["drift"], wildcards=ssl.one.wildcards
                        )
                        drift_samples = ssl.samples2times(drift["times"], direction="reverse")

                        # code from pyks. get the centers of the bins that they used
                        nblocks = (drift["um"].shape[1] + 1) // 2
                        yl = np.floor(geom[:, 1].max() / nblocks).astype("int") - 1
                        mins = np.linspace(0, geom[:, 1].max() - yl - 1, 2 * nblocks - 1)
                        maxs = mins + yl
                        centers = (mins + maxs) / 2
                        print(f"{centers.shape=} {drift_samples.shape=} {drift['um'].shape=}")
                        ksme = mu.NonrigidMotionEstimate(
                            -drift["um"].T,
                            time_bin_centers_s=drift_samples / fs,
                            spatial_bin_centers_um=centers,
                        )
                        save_motion_est(sessdir, "ks", ksme)
                        z_reg_ks = ksme.correct_s(t, z)
                        h5.create_dataset("z_reg_ks", data=z_reg_ks)

                shutil.copy(sub_h5, sessdir)
            except Exception as e:
                with open(sessdir / "metadata.pkl", "wb") as sess_jar:
                    metadata["subtraction_error"] = e
                    pickle.dump(metadata, sess_jar)
                shutil.rmtree(dscache)
                continue
            finally:
                shutil.rmtree(subcache)

        if args.ksreloc and not (sessdir / "ks_relocalizations.npz").exists():
            print("Relocalizing KS")
            sl = SpikeSortingLoader(pid=pid, one=one)
            spikes, clusters, channels = sl.load_spike_sorting()
            clusters = sl.merge_clusters(spikes, clusters, channels)
            spike_times = spikes["times"]
            spike_frames = sl.samples2times(
                spike_times, direction="reverse"
            ).astype("int")
            spike_train = np.c_[spike_frames, spikes["clusters"]]

            # -- align kilosort spikes
            (
                aligned_spike_train,
                order,
                templates_aligned,
                template_shifts,
            ) = clean_align_and_get_templates(
                spike_train,
                rec.get_num_channels(),
                rec.get_binary_description()["file_paths"][0],
            )
            aligned_spike_index = aligned_spike_train.copy()
            for j in np.unique(aligned_spike_train[:, 1]):
                aligned_spike_index[aligned_spike_train[:, 1] == j, 1] = (
                    templates_aligned[j].ptp(0).argmax()
                )

            # -- localize kilosort spikes
            localizations, maxptp = grab_and_localize(
                aligned_spike_index,
                rec.get_binary_description()["file_paths"][0],
                rec.get_channel_locations(),
                n_jobs=args.njobs,
                chunk_size=int(np.floor(args.batchlen * fs)),
            )

            # -- save outputs
            np.savez(
                sessdir / "ks_relocalizations.npz",
                aligned_spike_train=aligned_spike_train,
                aligned_spike_index=aligned_spike_index,
                localizations=localizations,
                maxptp=maxptp,
            )

        with open(sessdir / "metadata.pkl", "wb") as sess_jar:
            metadata["done"] = True
            pickle.dump(metadata, sess_jar)
        shutil.rmtree(dscache)
