# spike-psvae

*Getting started.* Install `node` without conda, can't figure out how to avoid an environment conflict. Then,

```
conda create -n psvae python=3.8 numpy seaborn scikit-image scikit-learn scipy h5py tqdm
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

For jupyter lab, the way I'm doing it for now is:

```
pip install jupyterlab jupytext
jupyter lab build
```

Then serve:

```
mkdir -p .jupter/lab/workspaces
JUPYTERLAB_WORKSPACES_DIR=.jupyter/lab/workspaces jupyter lab --no-browser --ip=0.0.0.0
```


### To do:

*Done for now:*

 - [x] Test localization gives same result
    - Similar but not the same, same problems with y. Maybe here is matching ptps a little better.
 - [x] Full recentering analysis
    - [x] PCA / Tensor PCA
       - [x] Residuals in plots
    - [x] Cull collisions
    - [x] Write analysis pipeline function so it can be applied to multiple datasets
    - [x] What do `y==0` spikes look like when relocating?
    - [x] Run on culled templates and NP2 data
    - [ ] Run on NPUltra...
 - [x] More consistent geom? 18/22 channel version? Should not matter for now.
    - This is `geomkind="standard"` in the code. It puts the max channel z at the center, and then `channel_radius // 2` shanks above and below, for an odd number of shanks and `2 * channel_radius + 2` channels. The original geometry, which is called `geomkind="updown"` in the code, shifts `2 * channel_radius` channels up or down depending on how the ptp looks.
 - [x] Destriping reg

*Low priority:*

 - [ ] Get the denoiser / detection code running here, we'll need to ingest more data.
 - [ ] NPUltra standard geom.
 - [ ] Data augmentation
    - [ ] Spikes on the edge of the probe are outliers (maxchan is not near center) -- least squares doesn't care, but a neural net does. Maybe the pipeline can extract extra channels, and the data loader can randomly slide a window around. 
 - [ ] Heatmap vis cropping: can do this by using a non-constant dt - eg, show us samples at something like `[[-2:0.1:-1] [-2:0.02:-1] [1:0.1:2]]` ms in matlab notation, or something like this (dt changes per spike?)

*Working on:*

 - [ ] Train conv VAEs
    - Architecture code is here, but not learning well yet...
    - [ ] Learning stability: data processing, etc
       - [x] Probably should normalize input data somehow so that we don't have to lean on batchnorm so much. I think it is interfering with learning and we shouldn't need it...
       - [ ] Same for supervised keys -- they are not very Gaussian, might interfere with the DKL.
       - [ ] PCA preprocessing
       - [x] Scale the loss to standard units? Would correspond to like a N(0,n) or something
    - [x] Overfit [PS]VAE to templates
       - [ ] See if standard geom and z relative to bottom helps learning
    - [ ] Add in $\beta$ and all that, and figure out hyperparameter search. Hyperopt?
 - [x] Movies of PCA/Parafac features over time, do they drift?
    - [ ] If we still see wiggles in the clusters after re-locating, are the wiggles strongest during periods of large probe motion?
 - [ ] Relocate by interpolation / shift on Z rather than scaling by ptps
    - [x] Torch code to do image translation
    - [ ] How to deal with the boundary? Zeros? Or, could use the waveform on the edge and do PTP rescaling?
    - [ ] Grab larger local waveforms to avoid boundary problems when possible? Integrate relocation into the localization/denoising pipeline?
 - [ ] Clustering / final performance metrics
    - What is the minimum viable clustering that will produce a meaningful result? And, how best to compare them?
 - [ ] Downsample NPUltra and see if nonlinear interpolation helps upsample
 - [ ] Figure out what is up with ICP

### To do the week of Dec 6

 - [ ] Full analysis
   - [ ] Displacement maps
   - [x] Datoviz movies
      - at some point soon it will also be useful to see the multi-column version of this, with different columns showing z on the y axis and x,y,alpha,pc1,pc2 on the x axis in different columns (with z post-registration and the pcs post-relocation)
      - in the vids, pls reduce the excess black space...  also, might be useful to add y labels at some point so we can know which z values we're looking at?
         - does that mean we switch to 2d marker?
      - for the z-splits, probably useful to add the channel locations (in, say, orange) to the xy panel of the datoviz vid.  could maybe add these at (0,z) for the other panels?
   - [ ] Plot PCs. can we see the unrelocated pca basis next to the relocated pca basis?  curious how these are different
   - [ ] Scatter pair plots: zrel and PCS, zrel and PTP, est displacement for spike vs PCS+PTP
   - [ ] Generalized correlation scores for above
   - [ ] How many units does isosplit discover?

 - [ ] Get denoised (single channel) id1 np1 waveforms

 - [ ] Show kilosort labels


PCA diagnostic
 - to debug this, can you show this as a 20-panel fig (one per channel), with the first couple PCs shown in one fig and some sample denoised + relocated spikes shown in another fig? (edited) 
 - basically want to see some of the raw data that's going into the PCA and then the output of the PCA on the same scale, to make sure things look reasonable on each channel
 - and as always, would also be useful to see the original data next to the PCA projections and residuals


Analysis [input: denoised wfs]
 - Relocation: show PTPs, targets, etc before/after, and waveforms before/after
 - PCA before/after relocation
    - show PC waveforms
 - Datoviz movie
 - GCS z_disp vs PCs
 - Isosplit (after discard y=0)
    - How many units before/after reloc?
    - GCS PCs vs z_disp for each unit

### Bug tracker

 - [x] This thing with updown -> standard, then the predicted PTP is flipped. What??
 - [ ] Duplicate spikes. They have the same time in the spike index, but they can have different channels there. However, the `max_channels` value is the same. See indices 55, 56 in single channel NP2.


### Notes and questions

 - 40% of y are <0.1, to be excluded from net training?
 - Relocating to xz center could be tough for spikes on the edge of the probe, where the maxchan Z is far from the center of the local geom.
