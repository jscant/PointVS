"""
Convert from directory structure-based datasets to types-based datsets.
"""

import argparse
from pathlib import Path

from point_vs.utils import expand_path, load_yaml, ensure_writable


def directory_to_types(base_path):
    """Convert directory based datasets to one that uses types files.

    Arguments:
        base_path: directory containing two directories (receptors and ligands)
            and an optional rmsd_info.yaml file, which has serialised rmsd
            information for each structure

    Returns:
        String formatted according to the GNINA 1.0 types files, as specified
        by Koes et al. at https://github.com/gnina/gnina
    """

    def remove_base_path(p):
        return str(p).replace(str(base_path), '')[1:]

    def get_underscore_suffix(p):
        return Path(p).name.split('.')[0].split('_')[-1]

    types_string = ''
    base_path = expand_path(base_path)
    rmsd_info_yaml = base_path / 'rmsd_info.yaml'
    if rmsd_info_yaml.is_file():
        rmsd_info = load_yaml(rmsd_info_yaml)
    else:
        rmsd_info = None

    for lig_fname in Path(base_path, 'ligands').glob('**/*.parquet'):
        suffix = lig_fname.parent.name.split('_')[-1]
        rec_name = lig_fname.parent.name.split('_')[0]
        try:
            rec_fname = next((base_path / 'receptors').glob(
                '{0}*.parquet'.format(rec_name)))
        except StopIteration:
            raise RuntimeError(
                'Receptor for ligand {0} not found. Looking for file '
                'named {1}'.format(
                    lig_fname, rec_name + '.parquet'))
        label = 1 if suffix == 'actives' else 0
        if rmsd_info is not None:
            rmsd = rmsd_info[rec_name]['docked_wrt_crystal'][int(
                get_underscore_suffix(lig_fname))]
        else:
            rmsd = -1
        print(label, rmsd, remove_base_path(rec_fname),
              remove_base_path(lig_fname))
        types_string += '{0} {1} {2} {3}\n'.format(
            label,
            rmsd,
            remove_base_path(rec_fname),
            remove_base_path(lig_fname)
        )
    return types_string


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str,
                        help='Base directory containing the ligands and '
                             'receptors directories')
    parser.add_argument('--output_fname', '-o', type=str,
                        help='Name of output types file. Defaults to '
                             'the base path''s deepest directory name')
    args = parser.parse_args()
    output_fname = args.output_fname
    if output_fname is None:
        output_fname = Path(args.directory).name
    output_fname = expand_path(output_fname)
    ensure_writable(output_fname)
    types_string = directory_to_types(args.directory)
    if not output_fname.suffix:
        output_fname = output_fname.wit_suffix('.types')
    with open(output_fname, 'w') as f:
        f.write(types_string)
