import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pymol
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem.rdFMCS import FindMCS

from point_vs.attribution.attribution import load_model, download_pdb_file
from point_vs.attribution.attribution_fns import masking, cam
from point_vs.attribution.process_pdb import score_pdb
from point_vs.utils import mkdir, coords_to_string, PositionDict


class ScoreStruct:

    def __init__(self, coords, atomic_number, score):
        self.coords = coords
        self.score = score
        self.atomic_number = atomic_number

    def __str__(self):
        return '({0}): {1}'.format(self.coords, self.score)

    def __repr__(self):
        return self.__str__()


def merge_structures(*fnames, output_fname='merged.pdb'):
    """Merge multiple structures into one file.

    Any input or output format can be used, and multiple different input formats
    can be used together.

    Arguments:
        fnames: two or more input structures
        output_fname: the name of the merged output. The format is determined by
            the file extension provided here (default: pdb).
    """
    for idx, fname in enumerate(fnames):
        fname = str(Path(fname).expanduser())
        pymol.cmd.load(fname, str(idx))
    pymol.cmd.remove('resn hoh')
    pymol.cmd.remove('solvent')
    mkdir(output_fname.parent)
    pymol.cmd.save(str(Path(output_fname).expanduser()),
                   selection='({})'.format(
                       ' or '.join([str(i) for i in range(len(fnames))])))
    pymol.cmd.delete('all')


def score_structure(rec, lig, pdb_output_fname, model, model_args,
                    output_path, attribution_type='cam'):
    # Generate merged pdb file for attribution
    pdb_output_fname = Path(pdb_output_fname).expanduser()

    output_path = mkdir(output_path)
    merge_structures(rec, lig, output_fname=pdb_output_fname)
    attribution_fn = {'masking': masking, 'cam': cam}[attribution_type]

    return score_pdb(model, attribution_fn, pdb_output_fname, output_path,
                     model_args=model_args, only_process=['UNK'],
                     quiet=True)[0]


def extract_xyz_to_score_map(df):
    xs, ys, zs = df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()
    scores = df['attribution'].to_numpy()
    xyz_to_score = {}
    for idx, score in enumerate(scores):
        x, y, z = xs[idx], ys[idx], zs[idx]
        coords_str = coords_to_string((x, y, z))
        xyz_to_score[coords_str] = score
    return xyz_to_score


def constrained_attribution(model_weights, rec_pdb, lig_input_dir,
                            output_dir, attribution_type='cam', pdbid=None,
                            site=None, relative_to='pairwise'):
    model, model_kwargs, cmd_line_args = load_model(model_weights)
    path_to_xyz_scores = {}
    mols = {}
    pdb_output_dir = mkdir(output_dir)

    if relative_to == 'crystal':
        if site is None:
            raise RuntimeError(
                'Please provide a 3 letter site code when plotting relative to'
                'crystal structure')
        pdbpath = download_pdb_file(pdbid, output_dir)
        path_to_xyz_scores[':CRYSTAL:'] = extract_xyz_to_score_map(
            score_pdb(model, {'masking': masking, 'cam': cam}[attribution_type],
                      pdbpath, output_dir, model_args=cmd_line_args,
                      only_process=[site], quiet=True, save_ligand_sdf=True)[0])
        crystal_ligand_sdf = Path(output_dir, 'crystal_ligand.sdf')
        mols[':CRYSTAL:'] = Chem.SDMolSupplier(
            str(crystal_ligand_sdf), True, False)[0]

    for lig in Path(lig_input_dir).expanduser().glob('*.sdf'):
        print(lig)
        concat_pdb_fname = pdb_output_dir / (lig.name.split('.')[0] + '.pdb')
        path_to_xyz_scores[str(lig)] = extract_xyz_to_score_map(score_structure(
            rec=rec_pdb,
            lig=lig,
            pdb_output_fname=concat_pdb_fname,
            model=model,
            model_args=cmd_line_args,
            output_path=output_dir,
            attribution_type=attribution_type
        ))
        mols[str(lig)] = Chem.SDMolSupplier(str(lig), True, False)[0]

    maximum_common_substructure = Chem.MolFromSmarts(
        FindMCS(list(mols.values())).smartsString)

    scores = defaultdict(dict)
    for path in path_to_xyz_scores.keys():
        mol = mols[path]
        matches = mol.GetSubstructMatches(maximum_common_substructure)
        if len(matches) != 1:
            raise AttributeError(
                'There are {0} substructures matches for ligand {1}; there '
                'should be exactly one per ligand.'.format(len(matches),
                                                           path))
        indices = matches[0]
        conf = mol.GetConformer()
        xyz_to_scores = PositionDict(path_to_xyz_scores[path], eps=0.01)
        for substructure_idx, ligand_idx in enumerate(indices):
            pos = conf.GetAtomPosition(ligand_idx)
            str_coords_mcs = coords_to_string((pos.x, pos.y, pos.z))
            atomic_number = mol.GetAtomWithIdx(ligand_idx).GetAtomicNum()
            score = xyz_to_scores[str_coords_mcs]
            scores[substructure_idx][path] = ScoreStruct(
                str_coords_mcs, atomic_number, score)

    return scores


def dist_vs_score(scores_dict, output_fname, title):
    atomic_numbers = {}
    scores = defaultdict(list)
    positions = defaultdict(list)
    relative_to_crystal = False
    for atom_idx, path_dict in scores_dict.items():
        for path, score_struct in path_dict.items():
            if path == ':CRYSTAL:':
                relative_to_crystal = True
                continue
            score, coords = score_struct.score, score_struct.coords
            scores[atom_idx].append(score)
            positions[atom_idx].append(
                np.array([float(i) for i in coords.split()]))
            atomic_numbers[atom_idx] = score_struct.atomic_number

    rows = (len(scores) // 4) + min(1, len(scores) % 4)
    fig, axes = plt.subplots(
        rows, 4, sharex=True, sharey=True, squeeze=True, figsize=(25, 5 * rows))
    axes = axes.reshape((-1, 1)).squeeze()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                    left=False, right=False)
    plt.xlabel(r'$||x_1 - x_2||_2$ (Angstroms)')
    plt.ylabel(r'$\Delta$ attribution score')

    delta_scores, distances = defaultdict(list), defaultdict(list)
    for idx, atom_idx in enumerate(scores.keys()):
        n = len(scores[atom_idx])
        ax = axes[atom_idx]
        if relative_to_crystal:
            for i in range(n):
                crystal_score_struct = scores_dict[atom_idx][':CRYSTAL:']
                delta_scores[atom_idx].append(
                    scores[atom_idx][i] -
                    crystal_score_struct.score)
                distances[atom_idx].append(
                    np.linalg.norm(
                        positions[atom_idx][i] -
                        np.array([float(i) for i in
                                  crystal_score_struct.coords.split()])))
        else:
            for i in range(n):
                for j in range(i + 1, n):
                    delta_scores[atom_idx].append(
                        abs(scores[atom_idx][i] - scores[atom_idx][j]))
                    distances[atom_idx].append(
                        np.linalg.norm(positions[atom_idx][i] -
                                       positions[atom_idx][j]))
        ax.scatter(distances[atom_idx], delta_scores[atom_idx], s=0.5, c='k')
        ax.set_title('Atom {0} (Atomic number {1})'.format(
            atom_idx, atomic_numbers[atom_idx]))

    plt.title(title)
    plt.savefig(Path(output_fname).expanduser())
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Saved model weights')
    parser.add_argument('receptor', type=str, help='Receptor structure file')
    parser.add_argument(
        'ligands', type=str, help='Directory containing ligand sdf files')
    parser.add_argument('output_dir', help='Directory for output')
    parser.add_argument(
        '--pdbid', type=str, help='PDBID of original crystal structure')
    parser.add_argument(
        '--site', type=str, help='3 letter ligand chain identifier (relative '
                                 'to crysal only)')
    parser.add_argument(
        '--title', '-t', required=False, default='', help='Title of graph')
    args = parser.parse_args()

    relative_to = 'crystal' if args.pdbid is not None else 'pairwise'

    df = constrained_attribution(
        args.model, args.receptor, args.ligands, args.output_dir,
        pdbid=args.pdbid, relative_to=relative_to, site=args.site)

    title = args.title
    graph_fname = 'output' if title == '' else title.lower().replace(' ', '_')
    graph_fname += '.png'

    dist_vs_score(df, graph_fname, title)