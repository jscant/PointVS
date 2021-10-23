import argparse
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from atom_types import Typer
from point_vs.utils import mkdir, expand_path, no_return_parallelise


def get_type_map(types: list = None):
    t = Typer()

    if types is None:
        types = [
            ['AliphaticCarbonXSHydrophobe'],
            ['AliphaticCarbonXSNonHydrophobe'],
            ['AromaticCarbonXSHydrophobe'],
            ['AromaticCarbonXSNonHydrophobe'],
            ['Nitrogen', 'NitrogenXSAcceptor'],
            ['NitrogenXSDonor', 'NitrogenXSDonorAcceptor'],
            ['Oxygen', 'OxygenXSAcceptor'],
            ['OxygenXSDonor', 'OxygenXSDonorAcceptor'],
            ['Sulfur', 'SulfurAcceptor'],
            ['Phosphorus']
        ]
    out_dict = {}
    generic = []
    for i, element_name in enumerate(t.atom_types):
        for types_list in types:
            if element_name in types_list:
                out_dict[i] = types.index(types_list)
                break
        if not i in out_dict.keys():
            generic.append(i)

    generic_type = len(types)
    for other_type in generic:
        out_dict[other_type] = generic_type
    return out_dict


def gninatypes_to_parquet(input_filename, output_filename, struct_type):
    coords = []
    types = []
    bp_int = 1 if struct_type == 'receptor' else 0
    n_atom_types = 14
    with open(input_filename, 'rb') as f:
        size = struct.calcsize("fffi")
        bainfo = f.read(size)
        while bainfo != b'':
            ainfo = struct.unpack("fffi", bainfo)
            coords.append(ainfo[:-1])
            type_int = ainfo[-1] + (bp_int * n_atom_types)
            types.append(type_int)
            bainfo = f.read(size)

    types = np.array(types)
    coords = np.array(coords)

    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df['types'] = types
    df['bp'] = bp_int
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_filename)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('base_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('structure_type', type=str)

    args = parser.parse_args()
    assert args.structure_type in ('receptor', 'ligand'), \
        'structure_type must be either receptor or ligand'

    base_path = Path(args.base_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    output_dir = mkdir(output_dir)
    input_dir = expand_path(args.base_path)

    input_fnames, output_fnames = [], []
    for gt in input_dir.glob('**/*.gninatypes'):
        input_fnames.append(str(gt))
        output_fnames.append(str(output_dir / gt))
    print(input_fnames)
    print(output_fnames)
    no_return_parallelise(
        gninatypes_to_parquet, input_fnames, output_fnames, args.structure_type)
