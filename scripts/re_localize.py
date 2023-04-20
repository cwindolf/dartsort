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


sdsc_base_path = Path("/mnt/sdceph/users/ibl/data")

def eid2sdscpath(eid):
    pids, probes = one.eid2pid(eid)
    alyx_base_path = one.eid2path(eid)
    paths = {}
    for pid, probe in zip(pids, probes):
        rel_path = one.list_datasets(eid, "raw_ephys_data/probe00*ap.cbin")
        assert len(rel_path) == 1
        rel_path = Path(rel_path[0])
        searchdir = sdsc_base_path / p.relative_to(one.cache_dir) / rel_path.parent
        pattern = Path(rel_path.name).with_suffix(f".*.cbin")
        glob = list(searchdir.glob(str(pattern)))
        assert len(glob) == 1
        paths[probe] = glob[0]
        assert paths[probe].exists()

    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("minix", type=int)
    ap.add_argument("maxix", type=int, default=None)

    args = ap.parse_args()
    
    import torch
    print(f"{torch.cuda.is_available()=}")
    
    # OneAlyx.setup(base_url='https://alyx.internationalbrainlab.org', make_default=True)

    # one = OneAlyx(
    #     base_url="https://alyx.internationalbrainlab.org",
    #     username="charlie.windolf",
    # )
    one = ONE()

    sessions_rep_site = one.alyx.rest('sessions', 'list', dataset_types='spikes.times', tag='2022_Q2_IBL_et_al_RepeatedSite')

    minix = args.minix
    maxix = args.maxix
    if maxix is None:
        maxix = len(sessions_rep_site)
    print(f"{minix=} {maxix=}")

    outdir = Path("/moto/stats/users/ciw2107/re_localizations")
    outdir.mkdir(exist_ok=True, parents=True)

    sicache = Path("/local/sicache")
    # ppxcache = Path("/local/ppxcache")
    subcache = Path("/local/subcache")

    for session in sessions_rep_site[minix:maxix]:
        sessdir = outdir / session['id']
        sessdir.mkdir(exist_ok=True)

        if (sessdir / "session.pkl").exists():
            with open(sessdir / "session.pkl", "rb") as sess_jar:
                meta = pickle.load(sess_jar)
                if "done" in meta and meta["done"]:
                    print(session['id'], "already done.")
                    assert (sessdir / "subtraction.h5").exists()
                    continue

        with open(sessdir / "session.pkl", "wb") as sess_jar:
            pickle.dump(session, sess_jar)

        print("-" * 50)
        print(session['id'])
        rec = si.read_cbin_ibl(
            session['id'],
            first_ap_stream,
            cache_folder="/local/sicache",
        )
        print(rec)
        fs = int(rec.get_sampling_frequency())

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


        ttt = rec.get_traces(start_frame=int(1000*fs), end_frame=int(1000*fs)+1000)
        print(f"{ttt.min()=} {ttt.max()=}")

        fig = plt.figure()
        plt.imshow(ttt.T, aspect="auto")
        plt.colorbar()
        # plt.show()
        fig.savefig(sessdir / "rawsnip.png", dpi=300)
        plt.close(fig)

        # if subcache.exists():
        #     shutil.rmtree(subcache)
        sub_h5 = subtract.subtraction(
            rec,
            out_folder=subcache,
            thresholds=[12, 10, 8, 6, 5],
            n_sec_pca=40,
            save_subtracted_tpca_projs=False,
            save_cleaned_tpca_projs=False,
            save_denoised_tpca_projs=False,
            n_jobs=14,
            loc_workers=1,
            overwrite=False,
        )
        # shutil.rmtree(ppxcache)
        shutil.copy(sub_h5, sessdir)
        shutil.rmtree(subcache)

        with open(sessdir / "session.pkl", "wb") as sess_jar:
            meta = session.copy()
            meta['done'] = True
            pickle.dump(meta, sess_jar)

