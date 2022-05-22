import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem

from point_vs.attribution.attribution import attribute, pdb_coords_to_identifier
from point_vs.constants import AA_TRIPLET_CODES, VDW_RADII
from point_vs.dataset_generation.types_to_parquet import StructuralFileParser
from point_vs.models.load_model import load_model
from point_vs.utils import expand_path, mkdir


def get_ligand_to_hbond_map(pdb, lig_name='LIG'):
    coords_to_atom_id = pdb_coords_to_identifier(pdb)
    pdb_parser = StructuralFileParser('receptor')
    mol = pdb_parser.read_file(pdb)[0]
    res = {}
    for atom in mol:
        if atom.OBAtom.GetResidue() is None or \
                atom.OBAtom.GetResidue().GetName() != lig_name or \
                atom.atomicnum == 1:
            continue

        # (x, y, z) -> A:XXXX:YYY:NAME
        identifier = find_identifier(coords_to_atom_id, atom.coords)
        if atom.OBAtom.IsHbondAcceptor():
            res[identifier] = 'hba'
        elif atom.OBAtom.IsHbondDonor():
            res[identifier] = 'hbd'
        else:
            res[identifier] = 'none'
    return res


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
        input_fnames, model_path, output_dir, ligand_name, layer=1,
        use_rank=False):
    """Use multiple protein-ligand structures to score protein atoms.

    The importance of each protein atom is assumed to be related to its mean
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
        ligand_name: residue name of the ligand (same across all inputs)
        layer: which layer from the GNN to take attention weights from for
            scoring
        use_rank: use ranked order of bond scores in place of raw bond scores

    Returns:
        pd.DataFrame with columns:
            atom_name: name of protein atom, with format
                CHAIN:RESIDUE_NUMBER:RESIDUE_NAME:ATOM_NAME, for example
                A:1021:PHE:O
            mean_score: mean GNN score of protein atom across all structures
            gnn_rank: rank from 0 to n of atom importance according to mean
                GNN attention score.
    """

    def find_lig_pharm(x):
        if x < 0:
            return 'hbd'
        elif x > 0:
            return 'hba'
        return 'none'

    def find_protein_atom(bond_identifier):
        id_1, id_2 = bond_identifier.split('-')
        if id_1.split(':')[-2] == ligand_name:
            return id_2
        elif id_2.split(':')[-2] == ligand_name:
            return id_1
        raise RuntimeError('Ligand triplet code not found in either atom.')

    def find_ligand_atom(bond_identifier):
        id_1, id_2 = bond_identifier.split('-')
        if id_1.split(':')[-2] == ligand_name:
            return id_1
        elif id_2.split(':')[-2] == ligand_name:
            return id_2
        raise RuntimeError('Ligand triplet code not found in either atom.')

    with open(expand_path(input_fnames), 'r') as f:
        fnames = [expand_path(fname.strip()) for fname in
                  f.readlines() if len(fname.strip())]
    _, model, _, _ = load_model(expand_path(model_path))

    processed_dfs = []
    prot_atom_to_max_lig_atom = defaultdict(list)
    for fname in fnames:
        lig_to_hbond_map = get_ligand_to_hbond_map(fname, lig_name=ligand_name)
        dfs = attribute(
            'edge_attention', model_path, expand_path(output_dir),
            input_file=fname, only_process=ligand_name, write_stats=False,
            gnn_layer=layer, write_pse=False, atom_blind=True,
            loaded_model=model,
            quiet=True
        )
        scores = []
        for site_code, (score, df, _, _) in dfs.items():
            if site_code.split(':')[1] != 'A':
                continue
            scores.append(score)
            df['protein_atom'] = df['bond_identifier'].apply(find_protein_atom)
            protein_atom_dfs = []
            for protein_atom in list(set(df['protein_atom'])):
                sub_df = df[df['protein_atom'] == protein_atom].reset_index(
                    drop=True)
                max_idx = np.argmax(sub_df['protein_atom'].to_numpy())
                max_row = sub_df.iloc[[max_idx]]
                max_score = max_row['bond_score'].to_numpy()[0]
                lig_pharm = lig_to_hbond_map[find_ligand_atom(
                    list(max_row['bond_identifier'])[0])]
                if lig_pharm == 'hba':
                    lig_pharm = max_score
                elif lig_pharm == 'hbd':
                    lig_pharm = -max_score
                else:
                    lig_pharm = 0
                sub_df = sub_df.groupby('protein_atom', as_index=False).max()
                sub_df['bond_score'] = max_score
                prot_atom_to_max_lig_atom[protein_atom].append(lig_pharm)
                protein_atom_dfs.append(sub_df)

            df = pd.concat(protein_atom_dfs)
            df.sort_values(by='bond_score', inplace=True, ascending=False)
            if use_rank:
                df['bond_score'] = range(len(df))
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
    concat_df['lig_pharm'] = concat_df.index.map(prot_atom_to_max_lig_atom)
    lig_pharm_scores = []
    for scores in concat_df.lig_pharm:
        lig_pharm_scores.append(np.sum(np.array(scores)))
    concat_df['lig_pharm'] = lig_pharm_scores
    concat_df['lig_pharm'] = concat_df['lig_pharm'].apply(find_lig_pharm)

    concat_df['mean'] = concat_df.mean(axis=1, numeric_only=True)
    result_df = pd.DataFrame({
        'protein_atom': concat_df.index,
        'mean_score': concat_df['mean'],
        'lig_pharm': concat_df['lig_pharm']
    })
    result_df.sort_values(
        by='mean_score', ascending=use_rank, inplace=True)
    result_df['gnn_rank'] = range(len(result_df))
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def scores_to_pharmacophore_df(reference_pdb, atom_scores, use_rank=False):
    """Map mean GNN atom scores to coordinates in reference structure.

    GNN scores are assigned to atoms in reference protein structure, which are
    then assigned a pharmacophore type depending on their identity and
    environment. The coordinates of these atoms can then be used to place
    pharmacophores with the correct identity in the correct place.

    Arguments:
        reference_pdb: name of the PDB file containing the reference protein
            structure
        atom_scores: pd.DataFrame with the fields protein_aton, mean_score,
            and gnn_rank (result of binding_events_to_ranked_protein_atoms
            function)
        use_rank: use ranked order of bond scores in place of raw bond scores

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

    def get_pharmacophore(atom, lig_pharm='none'):
        smina_type = pdb_parser.obatom_to_smina_type(atom)
        if smina_type in ('Oxygen', 'Nitrogen', 'Sulfur') or \
                smina_type.endswith('DonorAcceptor'):
            if lig_pharm == 'hba':
                return smina_type, 'hbd'
            elif lig_pharm == 'hbd':
                return smina_type, 'hba'
            return smina_type, 'none'
        if smina_type.endswith('Donor'):
            return smina_type, 'hbd'
        if smina_type.endswith('Acceptor'):
            return smina_type, 'hba'
        return smina_type, 'none'

    id_to_score_map = {atom: score for atom, score in zip(
        atom_scores['protein_atom'], atom_scores['mean_score'])}
    id_to_lig_pharm_map = {atom: lig_pharm for atom, lig_pharm in zip(
        atom_scores['protein_atom'], atom_scores['lig_pharm']
    )}

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

        # (x, y, z) -> A:XXXX:YYY:NAME
        identifier = find_identifier(coords_to_atom_id, atom.coords)

        # A:XXXX:YYY:NAME -> score
        score = id_to_score_map.get(identifier, (-1) ** use_rank * np.inf)

        # A:XXXX:YYY:NAME -> lig pharm
        lig_pharm = id_to_lig_pharm_map.get(identifier, 'none')

        smina_type, pharmacophore = get_pharmacophore(atom, lig_pharm)
        x, y, z = atom.coords
        xs.append(x)
        ys.append(y)
        zs.append(z)
        pharmacophores.append(pharmacophore)
        smina_types.append(smina_type)
        scores.append(score)
        vdw.append(VDW_RADII[atom.atomicnum])

    df = pd.DataFrame({
        'x': xs,
        'y': ys,
        'z': zs,
        'vdw_radius': vdw,
        'smina_type': smina_types,
        'pharmacophore': pharmacophores,
        'score': scores
    }).sort_values('score', ascending=use_rank)
    df.reset_index(inplace=True, drop=True)
    return df


def pharmacophore_df_to_mols(df, cutoff=0, include_donor_acceptors=False):
    """Convert pharmacophore DataFrame to RDKit pharmacophore mols.

    Arguments:
        df: pd.DataFrame containing the field x, y, z, pharmacophore and score
        cutoff: minimum mean GNN attention score for an atom to be considered
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
        filtered_df = df.copy()[
            df['pharmacophore'].isin(pharmacophore_type)].reset_index(drop=True)
        filtered_df.sort_values(
            by='score', ascending=False, inplace=True)
        filtered_df = filtered_df[:cutoff]
        filtered_df = filtered_df[filtered_df['score'] != np.inf]
        filtered_df = filtered_df[filtered_df['score'] != -np.inf]
        print(filtered_df)

        smiles = atom_type * len(filtered_df)
        mol = Chem.MolFromSmiles(smiles)
        conf = Chem.Conformer(mol.GetNumAtoms())

        vdw_str = []
        for idx, (_, row) in enumerate(filtered_df.iterrows()):
            conf.SetAtomPosition(idx, list(row[['x', 'y', 'z']]))
            vdw_str.append(str(row['vdw_radius']))

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
    parser.add_argument('--cutoff', '-c', type=int, default=7,
                        help='Take top-N scoring atoms as pharmacophores')
    parser.add_argument('--include_donor_acceptors', '-i', action='store_true',
                        help='Include donor-acceptors as both donors and '
                             'acceptors')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Which GNN layer to take attention weights from')
    parser.add_argument('--use_rank', '-r', action='store_true',
                        help='Use rank of bond score rather than raw bond score'
                             '.')

    args = parser.parse_args()
    rank_df = binding_events_to_ranked_protein_atoms(
        args.filenames, args.model, args.output_dir,
        args.ligand_residue_code.upper(), layer=args.layer,
        use_rank=args.use_rank)
    df = scores_to_pharmacophore_df(
        args.apo_protein, rank_df, use_rank=args.use_rank)
    print(df[:10])
    hba_mol, hbd_mol = pharmacophore_df_to_mols(df, cutoff=args.cutoff)
    output_dir = mkdir(args.output_dir)
    with Chem.SDWriter(str(output_dir / 'hba.sdf')) as w:
        w.write(hba_mol)
    with Chem.SDWriter(str(output_dir / 'hbd.sdf')) as w:
        w.write(hbd_mol)
