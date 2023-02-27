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

ap = argparse.ArgumentParser()

ap.add_argument("minix", type=int)
ap.add_argument("maxix", type=int, default=None)

args = ap.parse_args()


# OneAlyx.setup(base_url='https://alyx.internationalbrainlab.org', make_default=True)

# one = OneAlyx(
#     base_url="https://alyx.internationalbrainlab.org",
#     username="charlie.windolf",
# )
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
)

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
    
    first_ap_stream = next(sn for sn in se.IblStreamingRecordingExtractor.get_stream_names(session=session['id']) if sn.endswith(".ap"))

    print("-" * 50)
    print(session['id'])
    rec = si.read_ibl_streaming_recording(
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
    rec = si.zscore(rec, mode="mean+std", num_chunks_per_segment=100, margin_frames=100*fs)
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

