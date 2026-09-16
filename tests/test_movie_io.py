import os
import tempfile
import subprocess
import sys
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
import tifffile

from movie_io import list_movie_files, read_movie, resolve_input_path, movie_output_path, write_movie
from data_process import test_preprocess_lessMemoryNoTail_chooseOne


class MovieInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.movie = np.arange(8 * 10 * 12, dtype=np.uint16).reshape(8, 10, 12)

    def save_h5(self, name='movie.h5', key='images', data=None):
        path = os.path.join(self.temp.name, name)
        with h5py.File(path, 'w') as handle:
            handle.create_dataset(key, data=self.movie if data is None else data)
        return path

    def test_file_and_directory_discovery(self):
        path = self.save_h5()
        self.assertEqual(resolve_input_path(path, 'train'), path)
        self.assertEqual(resolve_input_path(self.temp.name), self.temp.name)
        self.assertEqual(resolve_input_path('./datasets'), os.path.join('./datasets', 'train'))
        self.assertEqual(list_movie_files(path), [path])
        self.assertEqual(list_movie_files(self.temp.name), [path])
        with self.assertRaises(FileNotFoundError):
            list_movie_files(os.path.join(self.temp.name, 'missing'))

    def test_h5_time_selection_and_axis_orders(self):
        for order in ('tyx', 'txy', 'ytx', 'yxt', 'xty', 'xyt'):
            stored = np.transpose(self.movie, tuple('tyx'.index(axis) for axis in order))
            path = self.save_h5(order + '.HDF5', data=stored)
            result = read_movie(path, h5_axis_order=order, frame_limit=5)
            np.testing.assert_array_equal(result, self.movie[:5])
            self.assertEqual(result.dtype, self.movie.dtype)

    def test_dataset_selection(self):
        path = self.save_h5(key='group/movie')
        np.testing.assert_array_equal(read_movie(path), self.movie)
        with h5py.File(path, 'a') as handle:
            handle.create_dataset('other', data=self.movie + 1)
        with self.assertRaisesRegex(ValueError, 'Specify --h5_dataset'):
            read_movie(path)
        np.testing.assert_array_equal(read_movie(path, 'group/movie'), self.movie)
        with self.assertRaisesRegex(ValueError, 'numeric 3D'):
            read_movie(path, 'missing')
        with h5py.File(path, 'a') as handle:
            handle.create_dataset('images', data=self.movie)
        np.testing.assert_array_equal(read_movie(path), self.movie)

    def test_tiff_and_separate_series(self):
        path = os.path.join(self.temp.name, 'stack.tiff')
        tifffile.imwrite(path, self.movie, photometric='minisblack')
        np.testing.assert_array_equal(read_movie(path, frame_limit=5), self.movie[:5])
        path = os.path.join(self.temp.name, 'separate.tif')
        with tifffile.TiffWriter(path) as handle:
            write = getattr(handle, 'write', None) or handle.save
            for frame in self.movie:
                write(frame, photometric='minisblack', contiguous=False)
        np.testing.assert_array_equal(read_movie(path), self.movie)

    def test_h5_preprocessing_matches_tiff(self):
        path = self.save_h5()
        tif_path = os.path.join(self.temp.name, 'movie.tif')
        tifffile.imwrite(tif_path, self.movie, photometric='minisblack')
        args = SimpleNamespace(
            datasets_path=path, datasets_folder=None, patch_x=4, patch_y=4,
            patch_t=4, gap_x=2, gap_y=2, gap_t=2, test_datasize=6, scale_factor=1
        )
        h5_result = test_preprocess_lessMemoryNoTail_chooseOne(args, 0)
        args.datasets_path = tif_path
        tif_result = test_preprocess_lessMemoryNoTail_chooseOne(args, 0)
        np.testing.assert_array_equal(h5_result[1], tif_result[1])
        self.assertEqual(h5_result[2], tif_result[2])
        self.assertEqual(h5_result[3:], tif_result[3:])
        self.assertEqual(h5_result[1].shape, (6, 10, 12))
        self.assertTrue(h5_result[0])

    def test_output_location_and_format_roundtrip(self):
        for extension in ('.tif', '.tiff', '.h5', '.hdf5'):
            source = os.path.join(self.temp.name, 'movie' + extension)
            result = movie_output_path(source)
            self.assertEqual(result, os.path.join(self.temp.name, 'movie_denosied' + extension))
            self.assertEqual(movie_output_path(source, ''), result)
            write_movie(result, self.movie)
            np.testing.assert_array_equal(read_movie(result), self.movie)
            self.assertEqual(read_movie(result).dtype, self.movie.dtype)
            custom_dir = os.path.join(self.temp.name, 'custom', 'nested')
            custom = movie_output_path(source, custom_dir)
            self.assertEqual(custom, os.path.join(custom_dir, 'movie_denosied' + extension))
            write_movie(custom, self.movie + 1)
            np.testing.assert_array_equal(read_movie(custom), self.movie + 1)
            write_movie(custom, self.movie)
            np.testing.assert_array_equal(read_movie(custom), self.movie)
        self.assertFalse(any(name.startswith('.srdtrans_') for name in os.listdir(self.temp.name)))

    def test_batch_skips_previous_outputs(self):
        source = self.save_h5()
        result = movie_output_path(source)
        write_movie(result, self.movie)
        model_result = movie_output_path(source, model_name='PFC.pth')
        write_movie(model_result, self.movie)
        self.assertEqual(list_movie_files(self.temp.name), [source])
        self.assertEqual(list_movie_files(result), [result])

    def test_output_uses_model_name(self):
        for extension in ('.tif', '.tiff', '.h5', '.hdf5'):
            source = os.path.join(self.temp.name, 'movie' + extension)
            expected = os.path.join(self.temp.name, 'movie_denosied_PFC' + extension)
            self.assertEqual(movie_output_path(source, model_name='PFC.pth'), expected)
            self.assertEqual(movie_output_path(source, model_name='PFC'), expected)
            self.assertEqual(movie_output_path(source, model_name=os.path.join('pth', 'PFC.pth')), expected)
            self.assertNotEqual(expected, movie_output_path(source, model_name='HPC.pth'))
            self.assertEqual(
                movie_output_path(source, model_name='cad_03hz'),
                os.path.join(self.temp.name, 'movie_denosied_cad_03hz' + extension)
            )

    def test_output_compression_filters(self):
        for extension in ('.tif', '.tiff'):
            path = os.path.join(self.temp.name, 'compressed' + extension)
            write_movie(path, self.movie)
            with tifffile.TiffFile(path) as handle:
                self.assertTrue(all(int(page.compression) in (8, 32946) for page in handle.pages))
            np.testing.assert_array_equal(read_movie(path), self.movie)
        path = os.path.join(self.temp.name, 'compressed.h5')
        write_movie(path, self.movie)
        with h5py.File(path, 'r') as handle:
            dataset = handle['images']
            self.assertTrue(dataset.shuffle)
            filters = dataset.id.get_create_plist()
            self.assertEqual(filters.get_nfilters(), 2)
            self.assertEqual(filters.get_filter(0)[0], h5py.h5z.FILTER_SHUFFLE)
            self.assertEqual(filters.get_filter(1)[0], 32015)
            self.assertEqual(filters.get_filter(1)[2], (3,))
            np.testing.assert_array_equal(dataset[:], self.movie)
        # A fresh process must also register the Zstd filter on the read path.
        subprocess.check_call([
            sys.executable, '-c',
            'import sys; import numpy as np; from movie_io import read_movie; '
            'np.testing.assert_array_equal(read_movie(sys.argv[1]), '
            'np.arange(8*10*12,dtype=np.uint16).reshape(8,10,12))', path
        ], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == '__main__':
    unittest.main()
