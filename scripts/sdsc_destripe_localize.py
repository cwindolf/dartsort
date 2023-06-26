# %%
import numpy as np
from one.api import ONE, OneAlyx
import spikeinterface.full as si
import spikeinterface.extractors as se
from spike_psvae import subtract
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import shutil
import argparse
import subprocess

# %%
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("minix", type=int)
    ap.add_argument("maxix", type=int, default=None)
    ap.add_argument("--njobs", type=int, default=20)
    ap.add_argument("--nspca", type=int, default=40)
    ap.add_argument("--batchlen", type=float, default=1)
    ap.add_argument("--locworkers", type=int, default=2)

    args = ap.parse_args()

    import torch
    print(f"{torch.cuda.is_available()=}")

    # %%
    # OneAlyx.setup(base_url='https://alyx.internationalbrainlab.org', make_default=True)

    # %%
    one = ONE()

    # %%
    sessions_rep_site = one.alyx.rest('sessions', 'list', dataset_types='spikes.times', tag='2022_Q2_IBL_et_al_RepeatedSite')
    sessions_rep_site = list(sorted(sessions_rep_site, key=lambda session: session['id']))

    minix = args.minix
    maxix = args.maxix
    if maxix is None:
        maxix = len(sessions_rep_site)
    print(f"{minix=} {maxix=}")

    # %%
    sdsc_base_path = Path("/mnt/sdceph/users/ibl/data")

    def eid2sdscpath(eid):
        pids, probes = one.eid2pid(eid)
        alyx_base_path = one.eid2path(eid)
        paths = {}
        for pid, probe in zip(pids, probes):
            rel_path = one.list_datasets(eid, f"raw_ephys_data/{probe}*ap.cbin")
            assert len(rel_path) == 1
            rel_path = Path(rel_path[0])
            searchdir = sdsc_base_path / alyx_base_path.relative_to(one.cache_dir) / rel_path.parent
            pattern = Path(rel_path.name).with_suffix(f".*.cbin")
            glob = list(searchdir.glob(str(pattern)))
            assert len(glob) == 1
            paths[probe] = pid, glob[0]
            assert paths[probe][1].exists()
        return paths


    # %%

    outdir = Path("/mnt/sdceph/users/cwindolf/re_ds_localizations")
    outdir.mkdir(exist_ok=True, parents=True)
    subcache = Path("/tmp/subcache")
    dscache = Path("/tmp/dscache")
    dscache.mkdir(exist_ok=True)

    # %%

    # -----------------------------------------
    for session in sessions_rep_site[minix:maxix]:

        paths = eid2sdscpath(session['id'])
        if "probe00" in paths:
            pid, cbin_path = paths["probe00"]
        else:
            print("No probe00, skipping for now")
            
        sessdir = outdir / f"pid{pid}"
        sessdir.mkdir(exist_ok=True)
        
        metadata = dict(
            probe="probe00",
            pid=pid,
            cbin_path=cbin_path,
            session=session,
        )
        
        if (sessdir / "metadata.pkl").exists():
            with open(sessdir / "metadata.pkl", "rb") as sess_jar:
                meta = pickle.load(sess_jar)
                if "done" in meta and meta["done"]:
                    print(session['id'], "already done.")
                    if not (sessdir / "subtraction.h5").exists():
                        raise ValueError("Inconsistency with the done and no h5 for\n", "meta:", sessdir / "metadata.pkl", "\n", metadata)

                    # -----------------------------------------
                    continue

        with open(sessdir / "metadata.pkl", "wb") as sess_jar:
            pickle.dump(metadata, sess_jar)

        ch_path = list(cbin_path.parent.glob("*ap*.ch"))
        assert len(ch_path) == 1
        ch_path = ch_path[0]
        meta_path = list(cbin_path.parent.glob("*ap*.meta"))
        assert len(meta_path) == 1
        meta_path = meta_path[0]

        print("-" * 50)
        print(session['id'], cbin_path)
        rec_cbin = si.read_cbin_ibl(str(cbin_path.parent), cbin_file=str(cbin_path), ch_file=str(ch_path), meta_file=str(meta_path))
        print(rec_cbin)
        fs = int(rec_cbin.get_sampling_frequency())
        
        # copy to temp dir
        dscache.mkdir(exist_ok=True)
        cbin_rel = cbin_path.stem.split(".")[0] + ("".join(s for s in cbin_path.suffixes if len(s) < 10))
        shutil.copyfile(cbin_path, dscache / cbin_rel)
        meta_rel = meta_path.stem.split(".")[0] + ("".join(s for s in meta_path.suffixes if len(s) < 10))
        shutil.copyfile(meta_path, dscache / meta_rel)
        ch_rel = ch_path.stem.split(".")[0] + ("".join(s for s in ch_path.suffixes if len(s) < 10))
        shutil.copyfile(ch_path, dscache / ch_rel)
        
        # run destriping
        destriped_cbin = dscache / f"destriped_{cbin_rel}"
        if not destriped_cbin.exists():
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
                    metadata['destriping_error'] = e
                    pickle.dump(metadata, sess_jar)
                if (dscache / f"destriped_{cbin_rel}").exists():
                    (dscache / f"destriped_{cbin_rel}").unlink()
                continue
        
        assert destriped_cbin.exists()
        
        rec = si.read_binary(destriped_cbin, rec_cbin.sampling_frequency, rec_cbin.get_num_channels(), dtype="float32")
        rec.set_probe(rec_cbin.get_probe(), in_place=True)
        fs = rec.get_sampling_frequency()

        ttt = rec.get_traces(start_frame=rec.get_num_samples() // 2, end_frame=rec.get_num_samples() // 2+1000)
        print(f"{ttt.min()=} {ttt.max()=}")

        # if subcache.exists():
        #     shutil.rmtree(subcache)

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
                loc_feature=("ptp", "peak"),
            )
            shutil.copy(sub_h5, sessdir)

            with open(sessdir / "metadata.pkl", "wb") as sess_jar:
                metadata['done'] = True
                pickle.dump(metadata, sess_jar)
        except Exception as e:
            with open(sessdir / "metadata.pkl", "wb") as sess_jar:
                metadata['subtraction_error'] = e
                pickle.dump(metadata, sess_jar)
        finally:
            shutil.rmtree(subcache)
            shutil.rmtree(dscache)
