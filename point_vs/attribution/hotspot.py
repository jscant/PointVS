import argparse

import pandas as pd
from rdkit import Chem

from point_vs.attribution.attribution import attribute, pdb_coords_to_identifier
from point_vs.dataset_generation.types_to_parquet import StructuralFileParser
from point_vs.models.load_model import load_model
from point_vs.utils import expand_path, mkdir

AA_TRIPLET_CODES = {'ILE', 'GLU', 'CYS', 'TRP', 'ALA', 'PRO', 'PHE', 'ASN',
                    'GLY', 'THR', 'ARG', 'MET', 'HIS', 'VAL', 'GLN', 'TYR',
                    'LYS', 'LEU', 'SER', 'ASP'}

VDW_RADII = {1: 1.1, 2: 1.4, 3: 1.82, 4: 1.53, 5: 1.92, 6: 1.7, 7: 1.55,
             8: 1.52, 9: 1.47, 10: 1.54, 11: 2.27, 12: 1.73, 13: 1.84, 14: 2.1,
             15: 1.8, 16: 1.8, 17: 1.75, 18: 1.88, 19: 2.75, 20: 2.31, 21: 2.15,
             22: 2.11, 23: 2.07, 24: 2.06, 25: 2.05, 26: 2.04, 27: 2.0,
             28: 1.97, 29: 1.96, 30: 2.01, 31: 1.87, 32: 2.11, 33: 1.85,
             34: 1.9, 35: 1.85, 36: 2.02, 37: 3.03, 38: 2.49, 39: 2.32,
             40: 2.23, 41: 2.18, 42: 2.17, 43: 2.16, 44: 2.13, 45: 2.1, 46: 2.1,
             47: 2.11, 48: 2.18, 49: 1.93, 50: 2.17, 51: 2.06, 52: 2.06,
             53: 1.98, 54: 2.16, 55: 3.43, 56: 2.68, 57: 2.43, 58: 2.42,
             59: 2.4, 60: 2.39, 61: 2.38, 62: 2.36, 63: 2.35, 64: 2.34,
             65: 2.33, 66: 2.31, 67: 2.3, 68: 2.29, 69: 2.27, 70: 2.26,
             71: 2.24, 72: 2.23, 73: 2.22, 74: 2.18, 75: 2.16, 76: 2.16,
             77: 2.13, 78: 2.13, 79: 2.14, 80: 2.23, 81: 1.96, 82: 2.02,
             83: 2.07, 84: 1.97, 85: 2.02, 86: 2.2, 87: 3.48, 88: 2.83,
             89: 2.47, 90: 2.45, 91: 2.43, 92: 2.41, 93: 2.39, 94: 2.4}


def find_identifier(coords_to_identifier, coords):
    """Find the unique PDB atom identifier given the atom's coordinates.

    The need for this function arises from the difficulty when using floating
    points as dictionary keys.

    Arguments:
        coords_to_identifier: dictionary mapping from (x, y, z) tuples of
            coordinates to the identifying PDB string, of the format
            CHAIN:RESIDUE_NUMBER:RESIDUE_NAME:ATOM_NAME, for example
            A:1021:PHE:O
        coords: tuple of three floats with atomic coordinates of desired atom

    Returns:
         Unique PDB identifier for atom with input coords.
    """
    if not isinstance(coords, str):
        coords = ':'.join([str(coord) for coord in coords])
    try:
        return coords_to_identifier[coords]
    except KeyError:
        x, y, z = coords.split(':')
        x_int, x_dec = x.split('.')
        y_int, y_dec = y.split('.')
        z_int, z_dec = z.split('.')
        max_dec = max(len(x_dec), len(y_dec), len(z_dec))
        x = x_int + '.' + x_dec + '0' * (max_dec - len(x_dec))
        y = y_int + '.' + y_dec + '0' * (max_dec - len(y_dec))
        z = z_int + '.' + z_dec + '0' * (max_dec - len(z_dec))
        for i in range(3):
            try:
                return coords_to_identifier[':'.join((x, y, z))]
            except KeyError:
                x += '0'
                y += '0'
                z += '0'
        raise KeyError('Cannot find coords ', coords)


def binding_events_to_ranked_protein_atoms(
        input_fnames, model_path, output_dir, only_process):
    """Use multiple protein-ligand structures to score protein atoms.

    The importance of each protein atom is assumed to be related to its median
    importance across multiple binding events with different ligands. Each
    atom's importance per structure is taken to the maximum GNN attention
    score across all its connected edges in that structure.

    Arguments:
        input_fnames: return-delimited text file containing filenames of
            PDB structures, which in turn should contain only ligands in the
            binding site of interest. All ligands should have the same residue
            name. The same protein should be present in each structure, although
            the conformation may vary; the residue numbering system should be
            identical within the binding site across sturctures where possible.
        model_path: location of pytorch saved GNN weights file
        output_dir: directory in which results should be stored
        only_process: residue name of the ligand (same across all inputs)

    Returns:
        pd.DataFrame with columns:
            atom_name: name of protein atom, with format
                CHAIN:RESIDUE_NUMBER:RESIDUE_NAME:ATOM_NAME, for example
                A:1021:PHE:O
            median_score: median GNN score of protein atom across all structures
            gnn_rank: rank from 0 to n of atom importance according to median
                GNN attention score.
    """

    def find_protein_atom(bond_identifier):
        id_1, id_2 = bond_identifier.split('-')
        if id_1.split(':')[-2] == only_process:
            return id_2
        elif id_2.split(':')[-2] == only_process:
            return id_1
        raise RuntimeError('Ligand triplet code not found in either atom.')

    with open(expand_path(input_fnames), 'r') as f:
        fnames = [expand_path(fname.strip()) for fname in
                  f.readlines() if len(fname.strip())]
    model, _, _ = load_model(expand_path(model_path))

    processed_dfs = []
    for fname in fnames:
        dfs = attribute(
            'edge_attention', model_path, expand_path(output_dir),
            input_file=fname, only_process=only_process, write_stats=False,
            gnn_layer=1, write_pse=False, atom_blind=True, loaded_model=model,
            quiet=True
        )

        scores = []
        for site_code, (score, df, _, _) in dfs.items():
            scores.append(score)
            df['protein_atom'] = df['bond_identifier'].apply(find_protein_atom)
            df = df.groupby('protein_atom', as_index=False).max()
            df.sort_values(by='bond_score', inplace=True, ascending=False)
            df.rename(
                columns={'bond_score': 'bond_score_' + str(fname)},
                inplace=True
            )
            for col in ('xtal_distance', 'bond_length_rank', 'bond_identifier',
                        'identifier_0', 'identifier_1', 'gnn_rank'):
                del df[col]
            df.reset_index(drop=True, inplace=True)
            processed_dfs.append(df)
        print('Completed file ', fname.name, ', with scores: ', ', '.join(
            ['{:.3f}'.format(score) for score in scores]), sep='')

    processed_dfs = [df.set_index('protein_atom') for df in processed_dfs]
    concat_df = processed_dfs[0].join(processed_dfs[1:])
    concat_df['median'] = concat_df.median(axis=1, numeric_only=True)
    result_df = pd.DataFrame({
        'protein_atom': concat_df.index,
        'median_score': concat_df['median']
    })
    result_df.sort_values(
        by='median_score', ascending=False, inplace=True)
    result_df['gnn_rank'] = range(len(result_df))
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def scores_to_pharmacophore_df(reference_pdb, atom_scores):
    """Map median GNN atom scores to coordinates in reference structure.

    GNN scores are assigned to atoms in reference protein structure, which are
    then assigned a pharmacophore type depending on their identity and
    environment. The coordinates of these atoms can then be used to place
    pharmacophores with the correct identity in the correct place.

    Arguments:
        reference_pdb: name of the PDB file containing the reference protein
            structure
        atom_scores: pd.DataFrame with the fields protein_aton, median_score,
            and gnn_rank (result of binding_events_to_ranked_protein_atoms
            function)

    Returns:
        pd.DataFrame containing the columns x, y, z, pharmacophore, smina_type
        and score, sorted by score, where:

            x, y and z are cartesian coodinates of the pharmacophores
            vdw is the van der waals radius of the original protein atom
            score is the attention score of the atom at that position
            smina_type is the sminatype of the atom in the reference structure
                (for example, AromaticCarbonXSHydrophobe)
            pharmacophore is one of 'hydrophobic', 'hbda', 'hba', 'hbd', 'none'
    """

    def get_pharmacophore(atom):
        smina_type = pdb_parser.obatom_to_smina_type(atom)
        if smina_type.endswith('XSHydrophobe'):
            return smina_type, 'hydrophobic'
        if smina_type.endswith('DonorAcceptor'):
            return smina_type, 'hbda'
        if smina_type.endswith('Donor'):
            return smina_type, 'hbd'
        if smina_type.endswith('Acceptor'):
            return smina_type, 'hba'
        if smina_type in ('Oxygen', 'Nitrogen', 'Sulfur'):
            if atom.OBAtom.IsAromatic():
                return smina_type, 'hydrophobic'
            return smina_type, 'hbda'
        return smina_type, 'none'

    id_to_score_map = {atom: score for atom, score in zip(
        atom_scores['protein_atom'], atom_scores['median_score'])}
    reference_pdb = str(expand_path(reference_pdb))
    coords_to_atom_id = pdb_coords_to_identifier(reference_pdb)
    pdb_parser = StructuralFileParser('receptor')
    mol = pdb_parser.read_file(reference_pdb)[0]
    xs, ys, zs = [], [], []
    scores, pharmacophores, smina_types, vdw = [], [], [], []
    for atom in mol:
        if atom.OBAtom.GetResidue() is None or \
                atom.OBAtom.GetResidue().GetName() not in AA_TRIPLET_CODES or \
                atom.atomicnum == 1:
            continue

        vdw.append(VDW_RADII[atom.atomicnum])
        identifier = find_identifier(coords_to_atom_id, atom.coords)
        score = id_to_score_map.get(identifier, 0)
        smina_type, pharmacophore = get_pharmacophore(atom)
        x, y, z = atom.coords
        xs.append(x)
        ys.append(y)
        zs.append(z)
        pharmacophores.append(pharmacophore)
        smina_types.append(smina_type)
        scores.append(score)

    df = pd.DataFrame({
        'x': xs,
        'y': ys,
        'z': zs,
        'vdw_radius': vdw,
        'smina_type': smina_types,
        'pharmacophore': pharmacophores,
        'score': scores
    }).sort_values('score', ascending=False)
    return df


def pharmacophore_df_to_mols(df, cutoff=0, include_donor_acceptors=False):
    """Convert pharmacophore DataFrame to RDKit pharmacophore mols.

    Arguments:
        df: pd.DataFrame containing the field x, y, z, pharmacophore and score
        cutoff: minimum median GNN attention score for an atom to be considered
            a pharmacophore
        include_donor_acceptors: include donor-acceptor atoms as both donors and
            acceptors

    Returns:
         Two RDKit molecules containing HBA and HBD atoms represented by
         phosphorous and iodine atoms respectively.
    """
    res = []
    included_pharmacophores = [['hba'], ['hbd']]
    if include_donor_acceptors:
        included_pharmacophores[0].append('hbda')
        included_pharmacophores[1].append('hbda')
    for atom_type, pharmacophore_type in zip(
            ('P', 'I'), included_pharmacophores):
        filtered_df = df.copy().loc[df['score'] > cutoff]
        filtered_df = filtered_df[
            filtered_df['pharmacophore'].isin(pharmacophore_type)].reset_index(
            drop=True)
        smiles = atom_type * len(filtered_df)
        mol = Chem.MolFromSmiles(smiles)
        conf = Chem.Conformer(mol.GetNumAtoms())

        vdw_str = []
        for idx, row in filtered_df.iterrows():
            conf.SetAtomPosition(idx, list(row[['x', 'y', 'z']]))
            vdw_str.append(str(row['vdw_radius'] / 100))

        conf.SetId(0)
        mol.AddConformer(conf)
        mol.SetProp('vdw', '\n'.join(vdw_str))
        res.append(mol)
    return tuple(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', type=str,
                        help='Return-delimited text file containing PDB '
                             'filenames')
    parser.add_argument('apo_protein', type=str,
                        help='Structure of protein for which pharmacophores '
                             'are generated')
    parser.add_argument('model', type=str,
                        help='Filename of pytorch saved model checkpoint')
    parser.add_argument('ligand_residue_code', type=str,
                        help='PDB residue triplet code for ligand')
    parser.add_argument('output_dir', type=str,
                        help='Location in which to store results')
    parser.add_argument('--cutoff', '-c', type=float, default=0.0,
                        help='Threshold for median attention score for atom '
                             'to be considered a pharacophore (between 0 and '
                             '1)')
    parser.add_argument('--include_donor_acceptors', '-i', action='store_true',
                        help='Include donor-acceptors as both donors and '
                             'acceptors')

    args = parser.parse_args()
    rank_df = binding_events_to_ranked_protein_atoms(
        args.filenames, args.model, args.output_dir,
        args.ligand_residue_code.upper())
    df = scores_to_pharmacophore_df(args.apo_protein, rank_df)
    print(df[:10])
    hba_mol, hbd_mol = pharmacophore_df_to_mols(df)
    output_dir = mkdir(args.output_dir)
    with Chem.SDWriter(str(output_dir / 'hba.sdf')) as w:
        w.write(hba_mol)
    with Chem.SDWriter(str(output_dir / 'hbd.sdf')) as w:
        w.write(hbd_mol)
