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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("minix", type=int)
    ap.add_argument("maxix", type=int, default=None)
    ap.add_argument("--njobs", type=int, default=20)
    ap.add_argument("--batchlen", type=float, default=0.5)

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

    outdir = Path("/mnt/sdceph/users/cwindolf/re_localizations")
    outdir.mkdir(exist_ok=True, parents=True)
    subcache = Path("/tmp/subcache")
    sicache = Path("/tmp/sicache")

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
                    assert (sessdir / "subtraction.h5").exists()

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
        rec = si.read_cbin_ibl(str(cbin_path.parent), cbin_file=str(cbin_path), ch_file=str(ch_path), meta_file=str(meta_path))
        print(rec)
        fs = int(rec.get_sampling_frequency())


        if not sicache.exists():
            rec = si.highpass_filter(rec)
            rec = si.phase_shift(rec)
            bad_channel_ids, channel_labels = si.detect_bad_channels(rec, num_random_chunks=100)
            print(f"{bad_channel_ids=}")
            rec = si.interpolate_bad_channels(rec, bad_channel_ids)
            rec = si.highpass_spatial_filter(rec)
            # we had been working with this before -- should switch to MAD,
            # but we need to rethink the thresholds
            rec = si.zscore(rec, mode="mean+std", num_chunks_per_segment=100, margin_frames=250*fs)
            print(rec)
            # /local is too small
            # rec = rec.save_to_folder(folder=ppxcache)

            ttt = rec.get_traces(start_frame=rec.get_num_samples() // 2, end_frame=rec.get_num_samples() // 2+1000)
            print(f"{ttt.min()=} {ttt.max()=}")

            fig = plt.figure()
            plt.imshow(ttt.T, aspect="auto")
            plt.colorbar()
            # plt.show()
            fig.savefig(sessdir / "rawsnip.png", dpi=300)
            plt.close(fig)

            rec = rec.save_to_folder("/tmp/sicache", n_jobs=20, chunk_size=rec.get_sampling_frequency())

        rec = si.read_binary_folder("/tmp/sicache")
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
                n_sec_pca=40,
                save_subtracted_tpca_projs=False,
                save_cleaned_tpca_projs=False,
                save_denoised_tpca_projs=False,
                n_jobs=args.njobs,
                loc_workers=2,
                overwrite=False,
                n_sec_chunk=args.batchlen,
            )
        except Exception as e:
            with open(sessdir / "metadata.pkl", "wb") as sess_jar:
                metadata['subtraction_error'] = e
                pickle.dump(metadata, sess_jar)

        shutil.copy(sub_h5, sessdir)
        shutil.rmtree(subcache)
        shutil.rmtree(sicache)

        with open(sessdir / "metadata.pkl", "wb") as sess_jar:
            metadata['done'] = True
            pickle.dump(metadata, sess_jar)
