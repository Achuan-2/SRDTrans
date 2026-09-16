"""Movie input shared by inference discovery and preprocessing (T, Y, X)."""
import os

import numpy as np
import tifffile as tiff


MOVIE_SUFFIXES = ('.tif', '.tiff', '.h5', '.hdf5')


def resolve_input_path(datasets_path, datasets_folder=None):
    if os.path.isfile(datasets_path):
        return datasets_path
    # Preserve the original no-argument datasets/train input.
    if datasets_folder is None:
        datasets_folder = 'train' if datasets_path == './datasets' else ''
    return os.path.join(datasets_path, datasets_folder) if datasets_folder else datasets_path


def list_movie_files(input_path):
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(MOVIE_SUFFIXES):
            raise ValueError('Unsupported movie format: {}'.format(input_path))
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError('Movie input does not exist: {}'.format(input_path))
    files = sorted(
        os.path.join(input_path, name) for name in os.listdir(input_path)
        if name.lower().endswith(MOVIE_SUFFIXES)
        and os.path.isfile(os.path.join(input_path, name))
    )
    if not files:
        raise ValueError('No TIFF or H5 movies found in {}'.format(input_path))
    return files


def read_movie(path, h5_dataset=None, h5_axis_order='tyx', frame_limit=None):
    """Read a grayscale stack; limit H5 reads to the requested time range."""
    if frame_limit is not None and frame_limit <= 0:
        raise ValueError('frame_limit must be positive')
    if path.lower().endswith(('.h5', '.hdf5')):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError('H5 input requires h5py; install it in your Python environment.') from exc
        order = h5_axis_order.lower()
        if len(order) != 3 or set(order) != set('tyx'):
            raise ValueError('h5_axis_order must be a permutation of tyx')
        with h5py.File(path, 'r') as handle:
            candidates = []

            def collect(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and obj.dtype.kind in 'uif':
                    candidates.append('/' + name)

            handle.visititems(collect)
            if h5_dataset:
                key = '/' + h5_dataset.lstrip('/')
                if key not in candidates:
                    raise ValueError('H5 dataset {} must be a numeric 3D movie in {}'.format(key, path))
            else:
                key = next((name for name in ('/images', '/data', '/mov') if name in candidates), None)
                if key is None:
                    if len(candidates) != 1:
                        raise ValueError('Specify --h5_dataset for {}. Numeric 3D datasets: {}'.format(path, candidates))
                    key = candidates[0]
            selection = [slice(None)] * 3
            selection[order.index('t')] = slice(0, frame_limit)
            movie = handle[key][tuple(selection)]
            movie = np.transpose(movie, tuple(order.index(axis) for axis in 'tyx'))
    elif path.lower().endswith(('.tif', '.tiff')):
        movie = tiff.imread(path)
        # Some exporters store each grayscale frame as a separate series.
        if movie.ndim == 2:
            with tiff.TiffFile(path) as handle:
                if len(handle.pages) > 1 and all(
                    page.shape == movie.shape and page.samplesperpixel == 1
                    and page.dtype == movie.dtype for page in handle.pages
                ):
                    movie = handle.asarray(key=range(len(handle.pages)))
        if frame_limit is not None:
            movie = movie[:frame_limit]
    else:
        raise ValueError('Unsupported movie format: {}'.format(path))
    if movie.ndim != 3 or any(size == 0 for size in movie.shape):
        raise ValueError('Expected a grayscale movie (T, Y, X), but {} has shape {}'.format(path, movie.shape))
    return movie
