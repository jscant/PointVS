from pathlib import Path

from pandas import DataFrame
from plip.basic import config
from plip.basic.supplemental import create_folder_if_not_exists, start_pymol
from plip.plipcmd import logger
from plip.structure.preparation import PDBComplex
from pymol import cmd

from point_vs.attribution.attribution_fns import masking, cam, node_attention, \
    edge_attention, edge_embedding_attribution
from point_vs.attribution.interaction_parser import \
    StructuralInteractionParser, \
    StructuralInteractionParserFast
from point_vs.attribution.plip_subclasses import \
    PyMOLVisualizerWithBFactorColouring, VisualizerDataWithMolecularInfo
from point_vs.utils import mkdir
import numpy as np

def visualize_in_pymol(
        model, attribution_fn, plcomplex, output_dir, model_args,
        gnn_layer=None, only_process=None, bonding_strs=[], write_pse=True,
        override_attribution_name=None, atom_blind=False, inverse_colour=False,
        pdb_file=None, coords_to_identifier=None):
    """Visualizes the given Protein-Ligand complex at one site in PyMOL.

    This function is based on the origina plip.visualization.vizualise_in_pymol
    function, with the added functinoality which uses a machine learning model
    to colour the receptor atoms by
    """

    triplet_code = plcomplex.uid.split(':')[0]
    if len(only_process) and triplet_code not in only_process:
        cmd.reinitialize()
        return None, None, None, None

    vis = PyMOLVisualizerWithBFactorColouring(plcomplex)

    #####################
    # Set everything up #
    #####################

    pdbid = plcomplex.pdbid
    lig_members = plcomplex.lig_members
    chain = plcomplex.chain
    if config.PEPTIDES:
        vis.ligname = 'PeptideChain%s' % plcomplex.chain
    if config.INTRA is not None:
        vis.ligname = 'Intra%s' % plcomplex.chain

    ligname = vis.ligname
    hetid = plcomplex.hetid

    metal_ids = plcomplex.metal_ids
    metal_ids_str = '+'.join([str(i) for i in metal_ids])

    ########################
    # Basic visualizations #
    ########################

    start_pymol(run=True, options='-pcq',
                quiet=not config.VERBOSE and not config.SILENT)
    vis.set_initial_representations()

    cmd.load(plcomplex.sourcefile)
    current_name = cmd.get_object_list(selection='(all)')[0]

    logger.debug(
        f'setting current_name to {current_name} and PDB-ID to {pdbid}')
    cmd.set_name(current_name, pdbid)
    cmd.hide('everything', 'all')
    if config.PEPTIDES:
        cmd.select(ligname, 'chain %s and not resn HOH' % plcomplex.chain)
    else:
        cmd.select(ligname, 'resn %s and chain %s and resi %s*' % (
            hetid, chain, plcomplex.position))
    logger.debug(
        f'selecting ligand for PDBID {pdbid} and ligand name {ligname}')
    logger.debug(
        f'resn {hetid} and chain {chain} and resi {plcomplex.position}')

    # Visualize and color metal ions if there are any
    if not len(metal_ids) == 0:
        vis.select_by_ids(ligname, metal_ids, selection_exists=True)
        cmd.show('spheres', 'id %s and %s' % (metal_ids_str, pdbid))

    # Additionally, select all members of composite ligands
    if len(lig_members) > 1:
        for member in lig_members:
            resid, chain, resnr = member[0], member[1], str(member[2])
            cmd.select(ligname, '%s or (resn %s and chain %s and resi %s)' % (
                ligname, resid, chain, resnr))

    cmd.show('sticks', ligname)
    cmd.color('myblue')
    cmd.color('myorange', ligname)
    cmd.util.cnc('all')
    if not len(metal_ids) == 0:
        cmd.color('hotpink', 'id %s' % metal_ids_str)
        cmd.hide('sticks', 'id %s' % metal_ids_str)
        cmd.set('sphere_scale', 0.3, ligname)
    cmd.deselect()

    vis.make_initial_selections()

    results_fname = Path(output_dir, '{0}_{1}_results.txt'.format(
        pdbid.upper(), '_'.join(
            [hetid, plcomplex.chain, plcomplex.position]))).expanduser()

    if pdb_file is not None:
        parser = StructuralInteractionParserFast(pdb_file)
    else:
        parser = StructuralInteractionParser()

    score, df, edge_indices, edge_scores = vis.colour_b_factors_pdb(
        model, attribution_fn=attribution_fn, parser=parser,
        gnn_layer=gnn_layer, results_fname=results_fname,
        model_args=model_args,
        only_process=only_process, pdb_file=pdb_file,
        coords_to_identifier=coords_to_identifier)

    if attribution_fn == edge_attention:
        if bonding_strs is None or not len(bonding_strs):
            keep_df = df.copy()
            keep_df.sort_values(by='bond_score', inplace=True, ascending=False)
            keep_df.reset_index(inplace=True)
            keep_df.drop(index=keep_df[keep_df.index > 8].index, inplace=True)
            max_bond_score = np.amax(keep_df['bond_score'].to_numpy())
            keep_df.drop(index=keep_df[keep_df['bond_score']
                                       < 0.25 * max_bond_score].index)
            bonding_strs = {b_id: dist for b_id, dist in zip(
                keep_df.bond_identifier, keep_df.bond_score)}

    # vis.show_hydrophobic()  # Hydrophobic Contacts
    resis = vis.show_hbonds(
        bonding_strs, atom_blind=atom_blind,
        inverse_colour=inverse_colour)  # Hydrogen Bonds
    # vis.show_halogen()  # Halogen Bonds
    # vis.show_stacking()  # pi-Stacking Interactions
    # vis.show_cationpi()  # pi-Cation Interactions
    # vis.show_sbridges()  # Salt Bridges
    # vis.show_wbridges()  # Water Bridges
    # vis.show_metal()  # Metal Coordination

    vis.refinements()
    if resis is not None and len(resis):
        cmd.select(
            'AdditionalInteractingResidues', 'resi ' + ' resi '.join(resis))
        cmd.show('sticks', 'AdditionalInteractingResidues')

    vis.zoom_to_ligand()
    vis.selections_cleanup()
    vis.selections_group()
    vis.additional_cleanup()

    if config.DNARECEPTOR:
        # Rename Cartoon selection to Line selection and change repr.
        cmd.set_name('%sCartoon' % plcomplex.pdbid, '%sLines' % plcomplex.pdbid)
        cmd.hide('cartoon', '%sLines' % plcomplex.pdbid)
        cmd.show('lines', '%sLines' % plcomplex.pdbid)

    if config.PEPTIDES:
        filename = "%s_PeptideChain%s" % (pdbid.upper(), plcomplex.chain)
        if config.PYMOL:
            vis.save_session(config.OUTPATH, override=filename)
    elif config.INTRA is not None:
        filename = "%s_IntraChain%s" % (pdbid.upper(), plcomplex.chain)
        if config.PYMOL:
            vis.save_session(config.OUTPATH, override=filename)
    elif write_pse:
        if override_attribution_name is None:
            attribution_fn_name = {
                masking: 'masking',
                cam: 'cam',
                node_attention: 'node_attention',
                edge_attention: 'edge_attention',
                edge_embedding_attribution: 'edges',
            }.get(attribution_fn, 'MD_distances')
        else:
            attribution_fn_name = override_attribution_name
        filename = "_".join(
            [attribution_fn_name, hetid, plcomplex.chain, plcomplex.position])
        vis.save_session(plcomplex.mol.output_path, override=filename)
    if config.PICS:
        vis.save_picture(config.OUTPATH, filename)

    cmd.reinitialize()
    if not isinstance(df, DataFrame):
        return None, None, None, None

    return score, df, edge_indices, edge_scores


def score_pdb(
        model, attribution_fn, pdbfile, outpath, model_args,
        only_process=None, quiet=False, save_ligand_sdf=False):
    mol = PDBComplex()
    outpath = str(Path(outpath).expanduser())
    pdbfile = str(Path(pdbfile).expanduser())
    mol.output_path = outpath

    mol.load_pdb(pdbfile, as_string=False)

    for ligand in mol.ligands:
        mol.characterize_complex(ligand)

    mkdir(outpath)

    complexes = [VisualizerDataWithMolecularInfo(mol, site) for site in sorted(
        mol.interaction_sets)
                 if not len(mol.interaction_sets[site].interacting_res) == 0]

    if save_ligand_sdf:
        ligand_outpath = str(Path(outpath, 'crystal_ligand.sdf'))
        complexes[0].ligand.molecule.write(
            format='sdf', filename=ligand_outpath, overwrite=True)
        print('Saved ligand as sdf to', ligand_outpath)

    def score_atoms(
            model, attribution_fn, plcomplex, model_args, only_process=None):

        vis = PyMOLVisualizerWithBFactorColouring(plcomplex)

        if config.PEPTIDES:
            vis.ligname = 'PeptideChain%s' % plcomplex.chain
        if config.INTRA is not None:
            vis.ligname = 'Intra%s' % plcomplex.chain

        parser = StructuralInteractionParser()
        score, df = vis.score_atoms(
            parser, only_process, model, attribution_fn, model_args=model_args,
            quiet=quiet)
        return df

    dfs = [score_atoms(
        model, attribution_fn=attribution_fn, plcomplex=plcomplex,
        model_args=model_args, only_process=only_process)
        for plcomplex in complexes]
    dfs = [df for df in dfs if df is not None]
    return dfs


def score_and_colour_pdb(
        model, attribution_fn, pdbfile, outpath, model_args, only_process=None,
        gnn_layer=None, bonding_strs=None, write_pse=True, atom_blind=False,
        override_attribution_name=None, inverse_colour=False, pdb_file=None,
        coords_to_identifier=None):
    mol = PDBComplex()
    outpath = str(Path(outpath).expanduser())
    pdbfile = str(Path(pdbfile).expanduser())
    mol.output_path = outpath
    mol.load_pdb(pdbfile, as_string=False)

    for ligand in mol.ligands:
        mol.characterize_complex(ligand)

    create_folder_if_not_exists(outpath)

    complexes = [VisualizerDataWithMolecularInfo(mol, site) for site in sorted(
        mol.interaction_sets)
                 if not len(mol.interaction_sets[site].interacting_res) == 0]

    dfs = {
        plcomplex.uid: visualize_in_pymol(
            model,
            attribution_fn=attribution_fn,
            output_dir=outpath,
            plcomplex=plcomplex,
            model_args=model_args,
            gnn_layer=gnn_layer,
            only_process=only_process,
            bonding_strs=bonding_strs,
            write_pse=write_pse,
            override_attribution_name=override_attribution_name,
            atom_blind=atom_blind,
            inverse_colour=inverse_colour,
            pdb_file=pdb_file,
            coords_to_identifier=coords_to_identifier
        )
        for plcomplex in complexes
    }

    return {lig_id: info for lig_id, info in dfs.items() if info[0] is not None}
