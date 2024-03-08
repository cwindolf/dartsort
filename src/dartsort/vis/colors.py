import numpy as np

try:
    from importlib.resources import files
except ImportError:
    try:
        from importlib_resources import files
    except ImportError:
        raise ValueError("Need python>=3.10 or pip install importlib_resources.")

data_dir = files("dartsort.pretrained")
glasbey1024_npz = data_dir.joinpath("glasbey1024.npz")

with np.load(glasbey1024_npz) as npz:
    glasbey1024 = npz["glasbey1024"]

gray = np.array([0.5, 0.5, 0.5])

__all__ = ["glasbey1024", "gray"]
