"""
Some basic helper functions for formatting time and sticking dataframes
together.
"""
import math
import multiprocessing as mp
import shutil
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt


def are_points_on_plane(p1, p2, p3, p4, eps=1e-6):
    def extend_arr(arr):
        return np.array(list(arr) + [1])

    m = np.vstack([
        extend_arr(p1),
        extend_arr(p2),
        extend_arr(p3),
        extend_arr(p4)
    ])
    return abs(np.linalg.det(m)) < eps


def pretify_dict(d, padding=5):
    max_key_len = max([len(str(key)) for key in d.keys()])
    line_len = max_key_len + padding
    s = ''
    for key, value in d.items():
        spaces = ' ' * (line_len - len(str(key)))
        s += '{0}:{1}{2}\n'.format(
            key, spaces, value
        )
    return s[:-1]


def save_yaml(d, fname):
    """Save a dictionary in yaml format."""
    with open(Path(fname).expanduser(), 'w') as f:
        yaml.dump(d, stream=f)


def load_yaml(fname):
    """Load a yaml dictionary"""
    with open(Path(fname).expanduser(), 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


def get_layer_shapes(model):
    """Return a list of the shapes of all linear layers in a model."""
    shapes = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            shapes.append(layer._parameters['weight'].shape)
    return shapes


class PositionSet(set):
    """Helper class for providing a set of coordinates with soft lookup.

    Keys should be space-separated strings of coordinates ('x y z'). The
    precision with which values are retrieved is specified by <eps> in the
    constructor. The L2 norm is used to measure distance between an
    unrecognised query and all of the keys in the dictionary. Any query more
    than <eps> from all keys will be considered outside of the set.
    """

    def __init__(self, coords_set=None, eps=1e-3):
        if coords_set is None:
            coords_set = set()
        set.__init__(self, coords_set)
        self.eps = eps

    def __contains__(self, key):
        if set.__contains__(self, key):
            return True
        return self.get_closest_atom(key)

    def get_closest_atom(self, coord_str):
        def extract_coords(s):
            return np.array([float(i) for i in s.replace(',', ' ').split()])

        coords = extract_coords(coord_str)
        for candidate in self:
            candidate_coords = extract_coords(candidate)
            dist = np.linalg.norm(coords - candidate_coords)
            if dist <= self.eps:
                return True
        return False


class PositionDict(dict):
    """Helper class for providing a soft coordinate lookup table.

    Keys should be space-separated strings of coordinates ('x y z'). Values
    can be anything. The precision with which values are retrieved is specified
    by <eps> in the constructor. The L2 norm is used to measure distance
    between an unrecognised query and all of the keys in the dictionary. Any
    query more than <eps> from all keys will raise a KeyError.
    """

    def __init__(self, coords_to_values_map=None, eps=1e-3):
        if coords_to_values_map is None:
            coords_to_values_map = {}
        dict.__init__(self, coords_to_values_map)
        self.eps = eps

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return self.get_closest_atom(key)

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def get_closest_atom(self, coord_str):
        def extract_coords(s):
            return np.array([float(i) for i in s.replace(',', ' ').split()])

        coords = extract_coords(coord_str)
        for candidate in self.keys():
            candidate_coords = extract_coords(candidate)
            dist = np.linalg.norm(coords - candidate_coords)
            if dist <= self.eps:
                return dict.__getitem__(self, candidate)

        raise KeyError('No atoms found within {0} Angstroms of query atom with '
                       'coords {1}'.format(self.eps, coord_str))


def ensure_writable(path):
    mkdir(path.parent)


def truncate_float(x, precision=3, as_str=False):
    """Return input x truncated to <precision> dp."""
    str_x = '{{:.{}f}}'.format(precision + 1).format(x)
    decimal_pos = str_x.find('.')
    if decimal_pos == -1:
        if as_str:
            return str_x
        return float(x)
    after_decimal_value = str_x[decimal_pos + 1:decimal_pos + precision + 1]
    res_str = str_x[:decimal_pos] + '.' + after_decimal_value
    if as_str:
        return res_str
    return float(res_str)


def coords_to_string(coords, precision=3, enforce_exact_decimal_places=True):
    """Return string representation of truncated coordinates."""

    def enforce_decimal_places(s):
        if not enforce_exact_decimal_places:
            return s
        curr_dp = len(s.split('.')[-1])
        return s + '0' * max(0, precision - curr_dp)

    def fmt(x):
        x = truncate_float(x, as_str=True)
        return enforce_decimal_places(x)

    return ' '.join([fmt(x) for x in coords])


def ensure_exact_coords(df, precision=3):
    df.x = df.x.apply(truncate_float, precision=precision)
    df.y = df.y.apply(truncate_float, precision=precision)
    df.z = df.z.apply(truncate_float, precision=precision)


def print_df(df):
    """Print pandas dataframe in its entirity (with no truncation)."""
    with pd.option_context('display.max_colwidth', None):
        with pd.option_context('display.max_rows', None):
            with pd.option_context('display.max_columns', None):
                print(df)


def no_return_parallelise(func, *args, cpus=-1):
    cpus = mp.cpu_count() if cpus == -1 else cpus
    indices_to_multiply = []
    iterable_len = 1
    args = list(args)
    for idx in range(len(args)):
        if not isinstance(args[idx], (tuple, list, types.GeneratorType)):
            indices_to_multiply.append(idx)
        elif iterable_len == 1:
            iterable_len = len(args[idx])
        elif iterable_len != len(args[idx]):
            raise ValueError('Iterable args must have the same length')
    for idx in indices_to_multiply:
        args[idx] = [args[idx]] * iterable_len

    inputs = list(zip(*args))
    with mp.Pool(processes=cpus) as pool:
        pool.starmap(func, inputs)


def _set_precision(precision):
    """Set global torch precision to either 'double' or 'float'."""
    if precision == 'double':
        torch.set_default_dtype(torch.float64)
        torch.set_default_tensor_type(torch.DoubleTensor)
    else:
        torch.set_default_dtype(torch.float32)
        torch.set_default_tensor_type(torch.FloatTensor)


def to_numpy(torch_tensor):
    """Switch from a torch tensor to a numpy array (on cpu)."""
    return torch_tensor.detach().cpu().numpy()


def mkdir(*paths):
    """Make a new directory, including parents."""
    path = Path(*paths).expanduser().resolve()
    path.mkdir(exist_ok=True, parents=True)
    return path


def expand_path(*paths):
    return Path(*paths).expanduser().resolve()


_use_gpu = False


def set_gpu_mode(mode):
    """Global usage of GPU."""
    global _use_gpu
    _use_gpu = mode
    if mode:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True


def condense(arr, gap=100):
    """Condense large arrays into averages over a given window size.

    Arguments:
        arr: numpy array or list of numbers
        gap: size of window over which to average array

    Returns:
        Tuple with new condensed counts (x) and smaller array (y) which is the
        mean of every <gap> values.
    """
    arr = np.array(arr)
    x = np.arange(0, len(arr), step=gap)
    y = np.array([np.mean(arr[n:n + gap]) for n in range(0, len(arr), gap)])
    return x, y


def get_eta(start_time, iters_completed, total_iters):
    """Format time in seconds to hh:mm:ss."""
    time_elapsed = time.time() - start_time
    time_per_iter = time_elapsed / (iters_completed + 1)
    time_remaining = max(0, time_per_iter * (total_iters - iters_completed - 1))
    formatted_eta = format_time(time_remaining)
    return formatted_eta


def format_time(t):
    """Returns string continaing time in hh:mm:ss format.

    Arguments:
        t: time in seconds

    Raises:
        ValueError if t < 0
    """
    if t < 0:
        raise ValueError('Time must be positive.')

    t = int(math.floor(t))
    h = t // 3600
    m = (t - (h * 3600)) // 60
    s = t - ((h * 3600) + (m * 60))
    return '{0:02d}:{1:02d}:{2:02d}'.format(h, m, s)


class Timer:
    """Simple timer class.

    To time a block of code, wrap it like so:

        with Timer() as t:
            <some_code>
        total_time = t.interval

    The time taken for the code to execute is stored in t.interval.
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def print_with_overwrite(*s, spacer=' '):
    """Prints to console, but overwrites previous output, rather than creating
    a newline.

    Arguments:
        s: string (possibly with multiple lines) to print
        spacer: whitespace character to use between words on each line
    """
    s = '\n'.join(
        [spacer.join([str(word) for word in substring]) for substring in s])
    erase = '\x1b[2K'
    up_one = '\x1b[1A'
    lines = s.split('\n')
    n_lines = len(lines)
    console_width = shutil.get_terminal_size((0, 20)).columns
    for idx in range(n_lines):
        lines[idx] += ' ' * max(0, console_width - len(lines[idx]))
    print((erase + up_one) * (n_lines - 1) + s, end='\r', flush=True)


def plot_with_smoothing(y, gap=100, figsize=(12, 7.5), ax=None):
    """Plot averages with a window given by <gap>."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    plt.cla()
    x, y = condense(y, gap=gap)
    ax.plot(x, y, 'k-')
    return ax
