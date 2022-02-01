import argparse
import multiprocessing as mp
import urllib
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd
from openbabel import openbabel
from plip.basic.supplemental import extract_pdbid

from point_vs.utils import mkdir, no_return_parallelise, coords_to_string, \
    PositionSet, expand_path

try:
    from openbabel import pybel
except (ModuleNotFoundError, ImportError):
    import pybel


def fetch_pdb(pdbid):
    """Modified plip function."""
    pdbid = pdbid.lower()
    pdburl = f'https://files.rcsb.org/download/{pdbid}.pdb'
    try:
        pdbfile = urlopen(pdburl).read().decode()
        if 'sorry' in pdbfile:
            print('No file in PDB format available from wwPDB for', pdbid)
            return None, None
    except HTTPError:
        print('No file in PDB format available from wwPDB for', pdbid)
        return None, None
    return [pdbfile, pdbid]


class Info:
    """Data structure to hold atom type data"""

    def __init__(
            self,
            sm,
            smina_name,
            adname,
            anum,
            ad_radius,
            ad_depth,
            ad_solvation,
            ad_volume,
            covalent_radius,
            xs_radius,
            xs_hydrophobe,
            xs_donor,
            xs_acceptor,
            ad_heteroatom,
    ):
        self.sm = sm
        self.smina_name = smina_name
        self.adname = adname
        self.anum = anum
        self.ad_radius = ad_radius
        self.ad_depth = ad_depth
        self.ad_solvation = ad_solvation
        self.ad_volume = ad_volume
        self.covalent_radius = covalent_radius
        self.xs_radius = xs_radius
        self.xs_hydrophobe = xs_hydrophobe
        self.xs_donor = xs_donor
        self.xs_acceptor = xs_acceptor
        self.ad_heteroatom = ad_heteroatom


class StructuralFileParser:

    def __init__(self, mol_type='ligand'):
        assert mol_type in ('ligand', 'receptor')
        self.mol_type = mol_type
        self.non_ad_metal_names = [
            "Cu",
            "Fe",
            "Na",
            "K",
            "Hg",
            "Co",
            "U",
            "Cd",
            "Ni",
            "Si",
        ]
        self.atom_equivalence_data = [("Se", "S")]
        self.atom_type_data = [
            Info(
                "Hydrogen",
                "Hydrogen",
                "H",
                1,
                1.000000,
                0.020000,
                0.000510,
                0.000000,
                0.370000,
                0.000000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "PolarHydrogen",
                "PolarHydrogen",
                "HD",
                1,
                1.000000,
                0.020000,
                0.000510,
                0.000000,
                0.370000,
                0.000000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "AliphaticCarbonXSHydrophobe",
                "AliphaticCarbonXSHydrophobe",
                "C",
                6,
                2.000000,
                0.150000,
                -0.001430,
                33.510300,
                0.770000,
                1.900000,
                True,
                False,
                False,
                False,
            ),
            Info(
                "AliphaticCarbonXSNonHydrophobe",
                "AliphaticCarbonXSNonHydrophobe",
                "C",
                6,
                2.000000,
                0.150000,
                -0.001430,
                33.510300,
                0.770000,
                1.900000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "AromaticCarbonXSHydrophobe",
                "AromaticCarbonXSHydrophobe",
                "A",
                6,
                2.000000,
                0.150000,
                -0.000520,
                33.510300,
                0.770000,
                1.900000,
                True,
                False,
                False,
                False,
            ),
            Info(
                "AromaticCarbonXSNonHydrophobe",
                "AromaticCarbonXSNonHydrophobe",
                "A",
                6,
                2.000000,
                0.150000,
                -0.000520,
                33.510300,
                0.770000,
                1.900000,
                False,
                False,
                False,
                False,
            ),
            Info(
                "Nitrogen",
                "Nitrogen",
                "N",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "NitrogenXSDonor",
                "NitrogenXSDonor",
                "N",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "NitrogenXSDonorAcceptor",
                "NitrogenXSDonorAcceptor",
                "NA",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                True,
                True,
                True,
            ),
            Info(
                "NitrogenXSAcceptor",
                "NitrogenXSAcceptor",
                "NA",
                7,
                1.750000,
                0.160000,
                -0.001620,
                22.449300,
                0.750000,
                1.800000,
                False,
                False,
                True,
                True,
            ),
            Info(
                "Oxygen",
                "Oxygen",
                "O",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "OxygenXSDonor",
                "OxygenXSDonor",
                "O",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "OxygenXSDonorAcceptor",
                "OxygenXSDonorAcceptor",
                "OA",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                True,
                True,
                True,
            ),
            Info(
                "OxygenXSAcceptor",
                "OxygenXSAcceptor",
                "OA",
                8,
                1.600000,
                0.200000,
                -0.002510,
                17.157300,
                0.730000,
                1.700000,
                False,
                False,
                True,
                True,
            ),
            Info(
                "Sulfur",
                "Sulfur",
                "S",
                16,
                2.000000,
                0.200000,
                -0.002140,
                33.510300,
                1.020000,
                2.000000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "SulfurAcceptor",
                "SulfurAcceptor",
                "SA",
                16,
                2.000000,
                0.200000,
                -0.002140,
                33.510300,
                1.020000,
                2.000000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "Phosphorus",
                "Phosphorus",
                "P",
                15,
                2.100000,
                0.200000,
                -0.001100,
                38.792400,
                1.060000,
                2.100000,
                False,
                False,
                False,
                True,
            ),
            Info(
                "Fluorine",
                "Fluorine",
                "F",
                9,
                1.545000,
                0.080000,
                -0.001100,
                15.448000,
                0.710000,
                1.500000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Chlorine",
                "Chlorine",
                "Cl",
                17,
                2.045000,
                0.276000,
                -0.001100,
                35.823500,
                0.990000,
                1.800000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Bromine",
                "Bromine",
                "Br",
                35,
                2.165000,
                0.389000,
                -0.001100,
                42.566100,
                1.140000,
                2.000000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Iodine",
                "Iodine",
                "I",
                53,
                2.360000,
                0.550000,
                -0.001100,
                55.058500,
                1.330000,
                2.200000,
                True,
                False,
                False,
                True,
            ),
            Info(
                "Magnesium",
                "Magnesium",
                "Mg",
                12,
                0.650000,
                0.875000,
                -0.001100,
                1.560000,
                1.300000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Manganese",
                "Manganese",
                "Mn",
                25,
                0.650000,
                0.875000,
                -0.001100,
                2.140000,
                1.390000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Zinc",
                "Zinc",
                "Zn",
                30,
                0.740000,
                0.550000,
                -0.001100,
                1.700000,
                1.310000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Calcium",
                "Calcium",
                "Ca",
                20,
                0.990000,
                0.550000,
                -0.001100,
                2.770000,
                1.740000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "Iron",
                "Iron",
                "Fe",
                26,
                0.650000,
                0.010000,
                -0.001100,
                1.840000,
                1.250000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            Info(
                "GenericMetal",
                "GenericMetal",
                "M",
                0,
                1.200000,
                0.000000,
                -0.001100,
                22.449300,
                1.750000,
                1.200000,
                False,
                True,
                False,
                True,
            ),
            # note AD4 doesn't have boron, so copying from carbon
            Info(
                "Boron",
                "Boron",
                "B",
                5,
                2.04,
                0.180000,
                -0.0011,
                12.052,
                0.90,
                1.920000,
                True,
                False,
                False,
                False,
            ),
        ]
        self.atom_types = [info.sm for info in self.atom_type_data]
        self.type_map = self.get_type_map()

    def get_type_map(self):
        """Original author: Constantin Schneider"""
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
            ['Phosphorus'],  # == 9
        ]
        out_dict = {}
        generic = []
        for i, element_name in enumerate(self.atom_types):
            for types_list in types:
                if element_name in types_list:
                    out_dict[i] = types.index(types_list)
                    break
            if i not in out_dict.keys():
                generic.append(i)

        generic_type = len(types)  # == 10
        for other_type in generic:
            out_dict[other_type] = generic_type
        return out_dict

    @staticmethod
    def read_file(infile, add_hydrogens=True):
        """Use openbabel to read in a pdb file.

        Original author: Constantin Schneider

        Args:
            infile (str): Path to input file
            add_hydrogens (bool): Add hydrogens to the openbabel OBMol object
        Returns:
            List of [pybel.Molecule]
        """
        molecules = []

        suffix = Path(infile).suffix[1:]
        file_read = pybel.readfile(suffix, str(infile))

        for mol in file_read:
            if add_hydrogens:
                mol.OBMol.AddHydrogens()
            molecules.append(mol)

        return molecules

    @staticmethod
    def adjust_smina_type(t, h_bonded, hetero_bonded):
        """Original author: Constantin schneider"""
        if t in ('AliphaticCarbonXSNonHydrophobe',
                 'AliphaticCarbonXSHydrophobe'):  # C_C_C_P,
            if hetero_bonded:
                return 'AliphaticCarbonXSNonHydrophobe'
            else:
                return 'AliphaticCarbonXSHydrophobe'
        elif t in ('AromaticCarbonXSNonHydrophobe',
                   'AromaticCarbonXSHydrophobe'):  # C_A_C_P,
            if hetero_bonded:
                return 'AromaticCarbonXSNonHydrophobe'
            else:
                return 'AromaticCarbonXSHydrophobe'
        elif t in ('Nitrogen', 'NitogenXSDonor'):
            # N_N_N_P, no hydrogen bonding
            if h_bonded:
                return 'NitrogenXSDonor'
            else:
                return 'Nitrogen'
        elif t in ('NitrogenXSAcceptor', 'NitrogenXSDonorAcceptor'):
            # N_NA_N_A, also considered an acceptor by autodock
            if h_bonded:
                return 'NitrogenXSDonorAcceptor'
            else:
                return 'NitrogenXSAcceptor'
        elif t in ('Oxygen' or t == 'OxygenXSDonor'):  # O_O_O_P,
            if h_bonded:
                return 'OxygenXSDonor'
            else:
                return 'Oxygen'
        elif t in ('OxygenXSAcceptor' or t == 'OxygenXSDonorAcceptor'):
            # O_OA_O_A, also an autodock acceptor
            if h_bonded:
                return 'OxygenXSDonorAcceptor'
            else:
                return 'OxygenXSAcceptor'
        else:
            return t

    def obatom_to_smina_type(self, ob_atom):
        """Original author: Constantin schneider"""
        atomic_number = ob_atom.atomicnum
        num_to_name = {1: 'HD', 6: 'A', 7: 'NA', 8: 'OA', 16: 'SA'}

        # Default fn returns True, otherwise inspect atom properties
        condition_fns = defaultdict(lambda: lambda: True)
        condition_fns.update({
            6: ob_atom.OBAtom.IsAromatic,
            7: ob_atom.OBAtom.IsHbondAcceptor,
            16: ob_atom.OBAtom.IsHbondAcceptor
        })

        # Get symbol
        ename = openbabel.GetSymbol(atomic_number)

        # Do we need to adjust symbol?
        if condition_fns[atomic_number]():
            ename = num_to_name.get(atomic_number, ename)

        atype = self.string_to_smina_type(ename)

        h_bonded = False
        hetero_bonded = False
        for neighbour in openbabel.OBAtomAtomIter(ob_atom.OBAtom):
            if neighbour.GetAtomicNum() == 1:
                h_bonded = True
            elif neighbour.GetAtomicNum() != 6:
                hetero_bonded = True

        return self.adjust_smina_type(atype, h_bonded, hetero_bonded)

    def string_to_smina_type(self, string: str):
        """Convert string type to smina type.

        Original author: Constantin schneider

        Args:
            string (str): string type
        Returns:
            string: smina type
        """
        if len(string) <= 2:
            for type_info in self.atom_type_data:
                # convert ad names to smina types
                if string == type_info.adname:
                    return type_info.sm
            # find equivalent atoms
            for i in self.atom_equivalence_data:
                if string == i[0]:
                    return self.string_to_smina_type(i[1])
            # generic metal
            if string in self.non_ad_metal_names:
                return "GenericMetal"
            # if nothing else found --> generic metal
            return "GenericMetal"

        else:
            # assume it's smina name
            for type_info in self.atom_type_data:
                if string == type_info.smina_name:
                    return type_info.sm
            # if nothing else found, return numtypes
            # technically not necessary to call this numtypes,
            # but including this here to make it equivalent to the cpp code
            return "NumTypes"

    def get_coords_and_types_info(
            self, mol, all_ligand_coords=None, add_polar_hydrogens=True):
        xs, ys, zs, atomic_nums, types, bp = [], [], [], [], [], []
        max_types_value = max(self.type_map.values()) + 2
        for atom in mol:
            if atom.OBAtom.GetResidue().GetName().lower() == 'hoh':
                continue
            atomic_num = atom.atomicnum
            if atomic_num == 1:
                if atom.OBAtom.IsNonPolarHydrogen() or not add_polar_hydrogens:
                    continue
                else:
                    type_int = max(self.type_map.values()) + 1
            else:
                smina_type = self.obatom_to_smina_type(atom)
                if smina_type == "NumTypes":
                    smina_type_int = len(self.atom_type_data)
                else:
                    smina_type_int = self.atom_types.index(smina_type)
                type_int = self.type_map[smina_type_int]
            if isinstance(all_ligand_coords, PositionSet):
                if coords_to_string(atom.coords) in all_ligand_coords:
                    bp.append(0)
                else:
                    type_int += max_types_value
                    bp.append(1)
            x, y, z = atom.coords
            xs.append(x)
            ys.append(y)
            zs.append(z)
            types.append(type_int)
            atomic_nums.append(atomic_num)
        return xs, ys, zs, types, atomic_nums, bp

    def obmol_to_parquet(self, mol, add_polar_hydrogens):
        xs, ys, zs, types, atomic_nums, _ = self.get_coords_and_types_info(
            mol, add_polar_hydrogens=add_polar_hydrogens)
        df = pd.DataFrame()
        df['x'], df['y'], df['z'] = xs, ys, zs
        df['atomic_number'] = atomic_nums
        df['types'] = types
        df['bp'] = int(self.mol_type == 'receptor')
        return df

    def file_to_parquets(
            self, input_file, output_path=None, output_fname=None,
            add_polar_hydrogens=True, sdf_idx=None):
        mols = self.read_file(input_file)
        if output_path is not None:
            output_path = mkdir(output_path)
        if output_fname is not None:
            output_fname = Path(output_fname)
        for idx, mol in enumerate(mols):
            if sdf_idx is not None and idx != sdf_idx:
                continue
            if output_fname is None:
                fname = Path(
                    mol.OBMol.GetTitle()).name.split('.')[0]
            else:
                fname = output_path / output_fname
                print(fname)
            df = self.obmol_to_parquet(mol, add_polar_hydrogens)
            if output_path is None:
                return df
            df.to_parquet(fname)

    def download_pdbs_from_csv(self, csv, output_dir):
        output_dir = Path(output_dir).expanduser()
        pdbids = set()
        with open(csv, 'r') as f:
            for line in f.readlines():
                pdbids.add(line.split(',')[0].lower())
        cpus = mp.cpu_count()
        inputs = [(pdbid, output_dir / pdbid) for pdbid in pdbids
                  if not Path(output_dir, pdbid, 'receptor.pdb').is_file()]
        with mp.get_context('spawn').Pool(processes=cpus) as pool:
            pool.starmap(self.download_pdb_file, inputs)

    @staticmethod
    def download_pdb_file(pdbid, output_dir):
        """Given a PDB ID, downloads the corresponding PDB structure.
        Checks for validity of ID and handles error while downloading.
        Returns the path of the downloaded file (From PLIP)"""
        output_dir = Path(output_dir).expanduser()
        pdbpath = output_dir / 'receptor.pdb'
        if pdbpath.is_file():
            print(pdbpath, 'already exists.')
            return
        if len(pdbid) != 4 or extract_pdbid(
                pdbid.lower()) == 'UnknownProtein':
            raise RuntimeError('Unknown protein ' + pdbid)
        while True:
            try:
                pdbfile, pdbid = fetch_pdb(pdbid.lower())
            except urllib.error.URLError:
                print('Fetching pdb {} failed, retrying...'.format(
                    pdbid))
            else:
                break
        if pdbfile is None:
            return 'none'
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(pdbpath, 'w') as g:
            g.write(pdbfile)
        print('File downloaded as', pdbpath)
        return pdbpath


def parse_types_file(types_file):
    def find_paths(line):
        recpath, ligpath = None, None
        chunks = line.split()
        for chunk in chunks:
            if chunk.find('.parquet') != -1:
                if recpath is None:
                    recpath = chunk
                else:
                    ligpath = chunk
                    break
        return recpath, ligpath

    recs, ligs = set(), set()
    with open(types_file, 'r') as f:
        for line in f.readlines():
            rec, lig = find_paths(line)
            recs.add(rec)
            ligs.add(lig)
    return list(recs), list(ligs)


def parse_single_types_entry(inp, outp, structure_type):
    def get_sdf_and_index(lig):
        sdf = '_'.join(str(lig).split('_')[:-1]) + '.sdf'
        idx = int(str(lig).split('_')[-1].split('.')[0])
        return sdf, idx

    def get_pdb(rec):
        return str(rec).replace(
            '.parquet', '.pdb').replace(
            '.gninatypes', '.pdb')

    parser = StructuralFileParser(structure_type)
    if structure_type == 'receptor':
        inp = get_pdb(inp)
        sdf_idx = None
    else:
        inp, sdf_idx = get_sdf_and_index(inp)
    parser.file_to_parquets(inp,
                            outp.parent,
                            outp.name.replace('.gninatypes', '.parquet'),
                            sdf_idx=sdf_idx)


def parse_types_mp(types_file, input_base_path, output_base_path):
    output_dir = mkdir(output_base_path)
    input_base_path = expand_path(input_base_path)
    recs, ligs = parse_types_file(types_file)
    inputs = recs + ligs
    structure_types = ['receptor' for _ in recs] + ['ligand' for _ in ligs]
    outputs = [Path(output_dir, input) for input in inputs]
    inputs = [Path(input_base_path, input) for input in inputs]
    no_return_parallelise(
        parse_single_types_entry, inputs, outputs, structure_types)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('types_file', type=str,
                        help='Input file, any of sdf, pdb or mol2 format '
                             'accepted')
    parser.add_argument('output_path', type=str,
                        help='Directory in which to store resultant parquet '
                             'files')
    parser.add_argument('input_base_path', type=str,
                        help='Root relative to which types file entries are '
                             'made. This should contain all of the SDF files '
                             'to be converted (the same as the argument '
                             '--base_path in generate_types_file.py).')
    args = parser.parse_args()

    parse_types_mp(args.types_file, Path(args.input_base_path).expanduser(),
                   args.output_path)
