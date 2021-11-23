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

*Low priority:*

 - [ ] Get the denoiser / detection code running here, we'll need to ingest more data.
 - [ ] NPUltra standard geom.
 - [ ] Data augmentation
    - [ ] Spikes on the edge of the probe are outliers (maxchan is not near center) -- least squares doesn't care, but a neural net does. Maybe the pipeline can extract extra channels, and the data loader can randomly slide a window around. 

*Working on:*

 - [ ] Train conv VAEs
    - Architecture code is here, but not learning well yet...
    - [ ] Overfit [PS]VAE to templates
       - [ ] See if standard geom and z relative to bottom helps learning
    - [ ] Add in $\beta$ and all that, and figure out hyperparameter search. Hyperopt?
 - [ ] Movies of PCA/Parafac features over time, do they drift?
    - If we still see wiggles in the clusters after re-locating, are the wiggles strongest during periods of large probe motion?
 - [ ] Heatmap vis cropping: can do this by using a non-constant dt - eg, show us samples at something like `[[-2:0.1:-1] [-2:0.02:-1] [1:0.1:2]]` ms in matlab notation, or something like this (dt changes per spike?)
 - [ ] Relocate by interpolation / shift on Z rather than scaling by ptps
 - [ ] Clustering / final performance metrics
    - What is the minimum viable clustering that will produce a meaningful result? And, how best to compare them?


### Bug tracker

 - [x] This thing with updown -> standard, then the predicted PTP is flipped. What??


### Notes and questions

 - 40% of y are <0.1, to be excluded from net training?
 - Relocating to xz center could be tough for spikes on the edge of the probe, where the maxchan Z is far from the center of the local geom.
