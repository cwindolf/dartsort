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

 - Test localization gives same result
 - Tensor PCA for recentered waveforms
 - Train conv VAEs
 - More consistent geom? 18/22 channel version? Should not matter for now.
