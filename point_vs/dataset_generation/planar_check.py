import argparse

from point_vs.preprocessing.pdb_to_parquet import StructuralFileParser
from point_vs.utils import expand_path, are_points_on_plane

try:
    from openbabel import pybel
except (ModuleNotFoundError, ImportError):
    import pybel


def is_planar_structure(fname, eps=1e-2):
    sfp = StructuralFileParser('receptor')
    mol = sfp.read_file(fname, add_hydrogens=False)[0]
    plane_defining_points = []
    for idx, atom in enumerate(mol):
        if atom.atomicnum == 1:
            continue
        coords = atom.coords
        if idx < 3:
            plane_defining_points.append(coords)
            continue
        if not are_points_on_plane(*plane_defining_points, coords, eps=eps):
            return False
    return True


def find_planar_structures(directory, eps=1e-2):
    planar_structures = []
    for idx, pdb in enumerate(expand_path(directory).glob('**/*.pdb')):
        if not (idx + 1) % 100:
            print('Processed', idx, 'structures')
        if is_planar_structure(pdb, eps=eps):
            planar_structures.append(pdb)
            print(pdb, 'is planar')
    return planar_structures


if __name__ == '__main__':
    pybel.ob.obErrorLog.SetOutputLevel(0)
    pybel.ob.obErrorLog.StopLogging()
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str,
                        help='Directory in which to search for planar PDB '
                             'structures')
    parser.add_argument('--eps', '-e', type=float, default=1e-1,
                        help='Parameter proportional to maximum deviation from '
                             'the plane')
    args = parser.parse_args()

    structures = find_planar_structures(args.directory, eps=args.eps)
    if not len(structures):
        print('No planar structures found')
    else:
        print('Planar structures:\n')
        for structure in structures:
            print(structure)
    with open('planar_pdbs.txt', 'w') as f:
        f.write('\n'.join([str(s) for s in structures]))
