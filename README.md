# dartsort

Install an environment for running this stuff with:

```bash
mamba create -n a python=3.10 numba seaborn scikit-learn scikit-image ipywidgets h5py colorcet tqdm joblib hdbscan cython jupyterlab jupytext ipywidgets widgetsnbextension nodejs nb_conda_kernels python-lsp-server black pyright jupyterlab-lsp
mamba activate a

# go to pytorch.org and find install instructions if you want gpu!
mamba install pytorch torchvision torchaudio -c pytorch

pip install matplotlib_venn

# For using jupyter
pip install jupyterlab-sublime jupyterlab-code-formatter

# optional
pip install ibllib

# definitely not optional :)
pip install -e ~/dartsort/
```
