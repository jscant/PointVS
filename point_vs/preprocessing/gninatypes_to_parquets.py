import argparse
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from atom_types import Typer
from joblib import Parallel, delayed


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


def _gninatypes_to_parquet(receptor, ligand, output_filename, type_map):
    bp_ints = []
    coords = []
    types = []
    n_atom_types = len(set(type_map.values()))
    for bp_int, gninatype in enumerate([receptor, ligand]):
        with open(gninatype, 'rb') as f:
            size = struct.calcsize("fffi")
            bainfo = f.read(size)
            while bainfo != b'':
                ainfo = struct.unpack("fffi", bainfo)
                coords.append(ainfo[:-1])
                type_int = type_map[ainfo[-1]] + (bp_int * n_atom_types)
                types.append(type_int)
                bp_ints.append(bp_int)
                bainfo = f.read(size)

    types = np.array(types)
    coords = np.array(coords)
    bp_ints = np.array(bp_ints)

    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df['types'] = types
    df['bp'] = bp_ints
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_filename)
    return 0


def gninatypes_to_parquet(
        input_filename, output_filename, type_map, struct_type):
    coords = []
    types = []
    bp_int = 1 if struct_type == 'receptor' else 0
    n_atom_types = len(set(type_map.values()))
    with open(input_filename, 'rb') as f:
        size = struct.calcsize("fffi")
        bainfo = f.read(size)
        while bainfo != b'':
            ainfo = struct.unpack("fffi", bainfo)
            coords.append(ainfo[:-1])
            type_int = type_map[ainfo[-1]] + (bp_int * n_atom_types)
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
    args = parser.parse_args()

    base_path = Path(args.base_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    type_map = get_type_map()
    output_dir.mkdir(parents=True, exist_ok=True)

    inf_outf = []
    for rec in (base_path / 'receptors').glob('*.gninatypes'):
        receptor = rec.stem.replace('_receptor', '')
        inf_outf.append(
            (rec, output_dir / 'receptors' / (receptor + '.parquet'),
             'receptor')
        )
        for ad in ['actives', 'decoys']:
            out_dir = receptor + '_' + ad
            for lig in (base_path / 'ligands' / out_dir).glob('*.gninatypes'):
                inf_outf.append((
                    lig,
                    output_dir / 'ligands' / out_dir / (lig.stem + '.parquet'),
                    'ligand'))

    with Parallel(n_jobs=12, verbose=5) as parallel:
        parallel(delayed(gninatypes_to_parquet)(
            inp, outp, type_map, struct_type) for (inp, outp, struct_type) in
                 inf_outf)
