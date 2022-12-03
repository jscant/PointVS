"""
Some basic helper functions for formatting time and sticking dataframes
together.
"""
import copy
import math
import multiprocessing as mp
import shutil
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem, SDMolSupplier, MolFromMol2File
from rdkit.Chem.rdMolAlign import CalcRMS
from scipy.stats import pearsonr

from point_vs import logging
from point_vs.constants import AA_TRIPLET_CODES


LOG = logging.get_logger('PointVS')


def find_latest_checkpoint(root, model_task=None):
    """Find latest saved checkpoint in directory."""
    if model_task is not None and model_task not in ('pose', 'affinity'):
        raise RuntimeError(
            'model_task must be either pose or affinity if specified.')
    if not model_task:
        model_task = ''
    glob_str = model_task + '*.pt'
    try:
        return max(Path(root, 'checkpoints').glob(glob_str),
                   key=lambda f: f.stat().st_ctime)
    except ValueError as exc:
        raise ValueError(f'No checkpoints found in {root}.') from exc

def flatten_nested_iterables(list_tup, unpack_arrays=False):
    """Flatten an arbitrarily deep nested list or tuple."""
    if isinstance(list_tup, (list, tuple)):
        if isinstance(list_tup[0], (list, tuple)):
            if len(list_tup) > 1:
                raise RuntimeError(
                    'Nested iterables have more than one iterable inside them.')
            return flatten_nested_iterables(list_tup[0], unpack_arrays)
        return list_tup[0]
    if isinstance(list_tup, (np.ndarray, torch.Tensor)) and unpack_arrays:
        return list_tup
    return list_tup


def get_n_cols(text_file):
    with open(expand_path(text_file), 'r', encoding='utf-8') as f:
        line = f.readline()
    return len(line.strip().split())


def split_sdfs(sdf, output_dir):
    cmd.reinitialize()
    if len(cmd.get_object_list()):
        raise RuntimeError('Pymol not properly wiped')
    sdf = expand_path(sdf)
    fname_base = sdf.with_suffix('').name
    cmd.load(str(sdf))
    cmd.split_states('all', prefix='state_')
    output_dir = mkdir(output_dir)
    output_sdfs = []
    for i in range(1, len(cmd.get_object_list())):
        mol_fname = '{0}_{1}.sdf'.format(fname_base, i - 1)
        mol_path = Path(output_dir, mol_fname)
        cmd.save(mol_path, selection='state_{}'.format(str(i).zfill(4)))
        output_sdfs.append(mol_path)
    return output_sdfs


def py_mollify(sdf, overwrite=False):
    """Use pymol to sanitise an SDF file for use in RDKit.

    Arguments:
        sdf: location of faulty sdf file
        overwrite: whether or not to overwrite the original sdf. If False,
            a new file will be written in the form <sdf_fname>_pymol.sdf

    Returns:
        Original sdf filename if overwrite == False, else the filename of the
        sanitised output.
    """
    sdf = Path(sdf).expanduser().resolve()
    mol2_fname = str(sdf).replace('.sdf', '_pymol.mol2')
    new_sdf_fname = sdf if overwrite else str(sdf).replace('.sdf', '_pymol.sdf')
    cmd.load(str(sdf))
    cmd.h_add('all')
    cmd.save(mol2_fname)
    cmd.reinitialize()
    cmd.load(mol2_fname)
    cmd.h_add('all')
    cmd.save(str(new_sdf_fname))
    return new_sdf_fname


def find_delta_E(sdf, multiple_structures=False):
    """
    Modified from https://github.com/bowenliu16/rl_graph_generation/blob
    /master/gym-molecule/gym_molecule/envs/molecule.py#L1131
    """
    original_sdf = sdf
    if multiple_structures:
        sdf = expand_path(sdf)
        split_dir = Path(
            sdf.parents[2],
            sdf.parents[1].name + '_split',
            sdf.parent.name
        )
        sdfs = split_sdfs(sdf, output_dir=split_dir)
    else:
        sdfs = [sdf]
    res = {}
    original_mols = {}
    original_energies = {}
    lowest_energy = np.inf
    lowest_energy_mol = None
    for idx, sdf in enumerate(sdfs):
        try:
            mol = SDMolSupplier(str(expand_path(sdf)))[0]
        except OSError:
            res[idx] = 'not_present'
            continue
        if mol is None:
            pymol_sdf = str(py_mollify(sdf))
            mol = SDMolSupplier(pymol_sdf)[0]
            if mol is not None:
                LOG.info(
                    f'Pymolification success for {Path(sdf).parent.name} (sdf)')
            else:
                mol = MolFromMol2File(pymol_sdf.replace('.sdf', '.mol2'))
                if mol is None:
                    res[idx] = 'pymol_fail'
                    continue
                else:
                    LOG.info(
                        f'Pymolification success for {Path(sdf).parent.name} (mol2)')
        if not idx in res.keys():
            Chem.AddHs(mol)
            original_mols[idx] = mol
            minimising_mol = copy.deepcopy(mol)
            if AllChem.MMFFHasAllMoleculeParams(mol):
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
                except:
                    res[idx] = 'forcefield_error'
                    continue
            else:
                res[idx] = 'unrecognised_atom_type'
                continue
            original_energy = ff.CalcEnergy()
            failed, opt_energy = AllChem.MMFFOptimizeMoleculeConfs(
                minimising_mol, maxIters=1000000, nonBondedThresh=1000)[0]
            if failed:
                res[idx] = 'did_not_converge'
            else:
                if opt_energy < lowest_energy:
                    lowest_energy = opt_energy
                    lowest_energy_mol = minimising_mol
                original_energies[idx] = original_energy

    for idx, mol in original_mols.items():
        if idx in res.keys():
            continue
        try:
            rmsd = CalcRMS(mol, lowest_energy_mol)
        except RuntimeError:
            res[idx] = 'no_common_substructure'
        else:
            res[idx] = (original_energies[idx] - lowest_energy, rmsd)

    return res


def get_regression_pearson(predictions_file):
    with open(expand_path(predictions_file), 'r') as f:
        if len(f.readlines()[0].split()) == 7:
            names = ('y_true', '_sep1', 'y_pred', 'receptor', 'ligand', '_set2',
                     'metric')
        else:
            names = ('y_true', '_sep1', 'y_pred', 'receptor', 'ligand')
    df = pd.read_csv(expand_path(predictions_file), sep='\s+',
                     names=names)
    return pearsonr(df['y_true'], df['y_pred'])


def rename_lig(fname, output_path=None, ligname='LIG', remove_solvent=True,
               handle_conformers='separate', atom_count_threshold=0,
               add_h=True):
    def detect_conformers():
        confs = set()
        with open(fname, 'r') as f:
            for line in f.readlines():
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                if len(line) > 16 and len(line[16].strip()):
                    confs.add(line[16])
        return confs

    def get_residue_ids():

        def obtain_res_id(chain, resi, resn):
            nonlocal res_ids
            res_ids.add((chain, resi, resn.strip().upper()))

        cmd.iterate('all',
                    'obtain_res_id(chain, resi, resn)',
                    space={'obtain_res_id': obtain_res_id})

    assert handle_conformers in ('separate', 'ignore', 'discard'), \
        'handle_conformers must be one of separate, ignore or discard'

    solvent_names = ('HOH', 'WAT', 'H20', 'TIP', 'SOL')  # default in pymol

    fname = str(expand_path(fname))
    fnames = []
    confs = detect_conformers()

    if handle_conformers == 'ignore' or len(confs) < 1:
        new_path = Path(
            mkdir(output_path), Path(fname).name.replace('.pdb', '_fixed.pdb'))
        shutil.copy(fname, new_path)
        fnames.append(new_path)
    else:
        for conf in confs:
            conf_fname = remove_conformers(
                fname, output_path, confs, conf, True,
                add_conf_suffix=handle_conformers == 'separate')
            fnames.append(conf_fname)
            if handle_conformers == 'discard':
                break
    fnames = sorted(fnames)
    for fname in fnames:
        cmd.reinitialize()
        cmd.load(fname)
        if remove_solvent:
            cmd.remove('solvent')

        if atom_count_threshold > 0:
            res_ids = set()
            get_residue_ids()
            for chain, res_id, res_name in res_ids:
                cmd.select(res_id, 'chain {0} and resi {1} and resn {2}'.format(
                    chain, res_id, res_name))
                if cmd.count_atoms(res_id) < atom_count_threshold and \
                        res_name not in solvent_names and res_name not in \
                        AA_TRIPLET_CODES:
                    cmd.remove(res_id)

        cmd.select('ligand', 'hetatm and not resn HOH')
        cmd.alter('ligand', 'resn="{}"'.format(ligname))
        if add_h:
            cmd.h_add('all')
        cmd.save(str(fname))
        cmd.remove('all')
        cmd.delete('all')
        cmd.reset()
        cmd.reinitialize()
    return fnames


def remove_conformers(
        fname, output_path, conf_ids, keep_id, add_fixed_suffix=False,
        add_conf_suffix=True):
    fname = expand_path(fname)
    cmd.reinitialize()
    cmd.load(str(fname))
    for conf_id in conf_ids:
        if conf_id == keep_id:
            continue
        selename = 'conf_' + conf_id
        cmd.select(selename, 'alt ' + conf_id)
        cmd.remove(selename)
    conf_suffix = '_conf_' + keep_id if add_conf_suffix else ''
    if add_fixed_suffix:
        outname = fname.with_suffix('').name + conf_suffix + '_fixed.pdb'
    else:
        outname = fname.with_suffix('').name + conf_suffix + keep_id + '.pdb'
    output_fname = Path(mkdir(output_path), outname)
    cmd.save(str(output_fname))
    cmd.remove('all')
    cmd.delete('all')
    cmd.reset()
    cmd.reinitialize()
    return output_fname


def fetch_atom_info(atom):
    """Fetch and assign relevant structural info to pybel atom."""
    obatom = atom.OBAtom
    res = obatom.GetResidue()
    atom.chain = res.GetChain().strip()
    atom_name = res.GetAtomID(obatom).strip()
    if len(atom_name) == 4:
        atom.name = atom_name[1:]
    else:
        atom.name = atom_name
    atom.resi = res.GetNum()
    atom.resn = res.GetName().strip()
    if not len(atom.chain):
        raise Exception('No chain for atom', atom)
    return '{0}:{1}:{2}:{3}'.format(
        atom.chain, atom.resi, atom.resn, atom.name)


def wipe_new_pdbs(directory, exempt=None):
    """Delete all .pdb files with exceptions, and prune empty directories.

    Arguments:
        directory: folder in which recurse and remove pdb files
        exempt: files and folders exempt from removal, usually the output of
            an earlier call to get_directory_state
    """

    def _rm_tree(pth):
        pth = Path(pth)
        for child in pth.glob('*'):
            if child.is_file() and child.suffix == '.pdb' and child not in \
                    exempt and not child.is_symlink():
                child.unlink()
            _rm_tree(child)
        if pth.is_dir() and not len(list(pth.glob('*'))) and \
                pth not in exempt and not pth.is_symlink():
            pth.rmdir()

    exempt = [] if exempt is None else [Path(p) for p in exempt]
    _rm_tree(directory)


def get_directory_state(directory):
    """Recursively return a list of all files and folders in a directory."""
    if not Path(directory).exists():
        return []
    if Path(directory).is_file():
        return [directory]
    directory_contents = list(Path(directory).glob('*'))
    if not len(directory_contents):
        return [directory]
    children = [directory]
    for item in directory_contents:
        children += get_directory_state(item)
    return children


def get_colour_interpolation_fn(c1, c2, min_val, max_val):
    """Generate a function which interpolates between RGB values.

    Arguments:
        c1: Bottom of the colour range for interpolation
        c2: Top of the colour range for interpolation
        min_val: Bottom of the range of values to interpolate between
        max_val: Top of the range of values to interpolate between

    Returns:
        Function which takes a value \in [min_value, max_value] and returns an
        numpy array with dimension (3,), the RGB value corresponding to a
        linear interpolation of the colours between c1 and c2 at the value
        given.

    Raises:
        AssertionError: all numbers in c2 must be greater or equal to those in
        c1
        AssertionError: max_val must be greater than or equal to min_val
    """
    c1, c2 = np.array(c1), np.array(c2)
    assert np.alltrue(c2 >= c1), 'All values in c2 must be <= those in c1'
    assert max_val >= min_val, 'max_val must be >= min_val'
    rgb_rng = c2 - c1
    val_rng = max_val - min_val

    def _interpolation_fn(val):
        # If there is only one value, give it c1
        if math.isclose(min_val, max_val):
            return list(c1)

        return list(np.array(c1) + ((val - min_val) / val_rng) * rgb_rng)

    return _interpolation_fn


def execute_cmd(cmd, raise_exceptions=True, silent=True):
    """Helper for executing obrms commands an capturing the output."""

    proc_result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True)
    if proc_result.stderr and raise_exceptions:
        LOG.exception(proc_result.stderr)
        raise subprocess.CalledProcessError(
            returncode=proc_result.returncode,
            cmd=proc_result.args,
            stderr=proc_result.stderr)
    res = proc_result.stdout.decode('utf-8')
    if proc_result.stdout and not silent:
        LOG.info(res)
    return res


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
    with open(Path(fname).expanduser(), 'w', encoding='utf-8') as f:
        yaml.dump(d, stream=f)


def load_yaml(fname):
    """Load a yaml dictionary"""
    # For backwards compatability reasons we should ignore missing constructors
    yaml.add_multi_constructor(
        '',
        lambda loader, suffix, node: None, Loader=yaml.SafeLoader)
    with open(Path(fname).expanduser(), 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def shorten_home(path, make_absolute=False):
    home_path = str(Path.home()) + '/'
    if make_absolute:
        path = expand_path(path)
    path = str(path)
    if path.startswith(home_path):
        return Path('~/' + path[len(home_path):])
    return Path(path)


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

        if isinstance(coord_str, (tuple, list)):
            coords = coord_str
        else:
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
                LOG.info(df)


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
    t = t or 0
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
