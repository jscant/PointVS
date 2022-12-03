"""Put PDB data through model to extract attribution scores."""
from pathlib import Path

import numpy as np
from pandas import DataFrame
from plip.basic import config
from plip.basic.supplemental import create_folder_if_not_exists, start_pymol
from plip.plipcmd import logger
from plip.structure.preparation import PDBComplex
from pymol import cmd

from point_vs import logging
from point_vs.utils import mkdir
from point_vs.utils import expand_path
from point_vs.attribution.attribution_fns import atom_masking
from point_vs.attribution.attribution_fns import cam
from point_vs.attribution.attribution_fns import edge_attention
from point_vs.attribution.attribution_fns import edge_embedding_attribution
from point_vs.attribution.attribution_fns import node_attention
from point_vs.attribution.attribution_fns import track_bond_lengths
from point_vs.attribution.attribution_fns import track_position_changes
from point_vs.attribution.attribution_fns import cam_wrapper
from point_vs.attribution.attribution_fns import attention_wrapper
from point_vs.attribution.attribution_fns import masking_wrapper
from point_vs.attribution.attribution_fns import bond_masking
from point_vs.attribution.interaction_parser import StructuralInteractionParser
from point_vs.attribution.interaction_parser import StructuralInteractionParserFast
from point_vs.attribution.plip_subclasses import PyMOLVisualizerWithBFactorColouring
from point_vs.attribution.plip_subclasses import VisualizerDataWithMolecularInfo


LOG = logging.get_logger('PointVS')


def visualize_in_pymol(
        model, attribution_fn, plcomplex, output_dir, model_args,
        gnn_layer=None, only_process=None, bonding_strs=None, write_pse=True,
        override_attribution_name=None, atom_blind=False, inverse_colour=False,
        pdb_file=None, coords_to_identifier=None, quiet=False, extended=False,
        split_by_mol=False):
    """Visualizes the given Protein-Ligand complex at one site in PyMOL.

    This function is based on the origina plip.visualization.vizualise_in_pymol
    function, with the added functinoality which uses a machine learning model
    to colour the receptor atoms by
    """

    if bonding_strs is None:
        bonding_strs = []

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

    naming_items = [item for item in
                    [hetid, plcomplex.chain, plcomplex.position] if
                    len(item.strip())]
    results_fname = Path(output_dir, '{0}_{1}_results.csv'.format(
        pdbid.upper(), '_'.join(naming_items))).expanduser()

    if pdb_file is not None:
        parser = StructuralInteractionParserFast(pdb_file, extended=extended)
    else:
        parser = StructuralInteractionParser(extended=extended)

    score, df, edge_indices, edge_scores = vis.colour_b_factors_pdb(
        model, attribution_fn=attribution_fn, parser=parser,
        gnn_layer=gnn_layer, results_fname=results_fname,
        model_args=model_args,
        only_process=only_process, pdb_file=pdb_file, split_by_mol=split_by_mol,
        coords_to_identifier=coords_to_identifier, quiet=quiet)

    if not write_pse:
        if not isinstance(df, DataFrame):
            return None, None, None, None

        return score, df, edge_indices, edge_scores

    if attribution_fn in (
            edge_attention, track_bond_lengths, attention_wrapper, cam_wrapper,
            bond_masking, masking_wrapper):
        if not len(bonding_strs):
            keep_df = df.copy()
            keep_df.sort_values(by='bond_score', inplace=True, ascending=False)
            keep_df.reset_index(inplace=True)
            keep_df.drop(index=keep_df[keep_df.index > 8].index, inplace=True)
            max_bond_score = np.amax(keep_df['bond_score'].to_numpy())
            keep_df.drop(index=keep_df[keep_df['bond_score']
                                       < 0.25 * max_bond_score].index)
            bonding_strs = {b_id: score for b_id, score in zip(
                keep_df.bond_identifier, keep_df.bond_score)}

    vis.show_hydrophobic()  # Hydrophobic Contacts
    resis = None
    resis = vis.show_hbonds(
        bonding_strs, atom_blind=atom_blind,
        inverse_colour=inverse_colour)  # Hydrogen Bonds
    vis.show_halogen()  # Halogen Bonds
    vis.show_hbonds_original()
    vis.show_stacking()  # pi-Stacking Interactions
    vis.show_cationpi()  # pi-Cation Interactions
    vis.show_sbridges()  # Salt Bridges
    vis.show_wbridges()  # Water Bridges
    vis.show_metal()  # Metal Coordination

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
                atom_masking: 'atom_masking',
                cam: 'cam',
                node_attention: 'node_attention',
                edge_attention: 'edge_attention',
                edge_embedding_attribution: 'edges',
                track_position_changes: 'displacement',
                track_bond_lengths: 'bond_lengths',
                attention_wrapper: 'attention',
                cam_wrapper: 'class_activation',
                bond_masking: 'bond_masking',
                masking_wrapper: 'masking'
            }.get(attribution_fn, 'MD_distances')
        else:
            attribution_fn_name = override_attribution_name
        naming_items = [item for item in [attribution_fn_name,
                                          hetid,
                                          plcomplex.chain,
                                          plcomplex.position] if
                        len(item.strip())]
        filename = "_".join(naming_items)
        vis.save_session(plcomplex.mol.output_path, override=filename)
        pth = expand_path(plcomplex.mol.output_path, filename + '.pse')
        LOG.info(f'Saved session as {pth}')
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
        LOG.info(f'Saved ligand as sdf to {ligand_outpath}.')

    def score_atoms(
            model, attribution_fn, plcomplex, model_args, only_process=None):

        vis = PyMOLVisualizerWithBFactorColouring(plcomplex)

        if config.PEPTIDES:
            vis.ligname = 'PeptideChain%s' % plcomplex.chain
        if config.INTRA is not None:
            vis.ligname = 'Intra%s' % plcomplex.chain

        score, df = vis.score_atoms(
            only_process, model, model_args, attribution_fn,
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
        coords_to_identifier=None, quiet=False, only_first=False,
        split_by_mol=False, extended=False):
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

    if only_first:
        complexes = complexes[:1]
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
            coords_to_identifier=coords_to_identifier,
            quiet=quiet,
            extended=extended,
            split_by_mol=split_by_mol,
        )
        for plcomplex in complexes
    }

    return {lig_id: info for lig_id, info in dfs.items() if info[0] is not None}
