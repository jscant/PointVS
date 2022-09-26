"""(Originally written for use by Steph Wills)

Take a trained model and a set of input sdf/pdb receptor and ligands and predict
the probability of a good pose or the binding affinity.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Union

from point_vs.dataset_generation.types_to_parquet import \
    StructuralFileParser
from point_vs.inference import get_model_and_test_dl
from point_vs.utils import expand_path, mkdir


Fname = Union[str, Path]


def generate_types_file(input_fnames: Fname, types_fname: Fname) -> None:
    """Generate a types file from the inputs, replacing extensions."""
    types = ''
    line_template = '{0} {1}\n'
    with open(input_fnames, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            chunks = line.split()
            if len(chunks) != 2:
                continue
            rec_pdb, lig_sdf = chunks
            rec_gt = rec_pdb.replace('.pdb', '.parquet')
            lig_gt = lig_sdf.replace(
                '.sdf', '.mol2').replace('.mol2', '.parquet')
            types += line_template.format(rec_gt, lig_gt)
    with open(expand_path(types_fname), 'w', encoding='utf-8') as f:
        f.write(types)

def predict_on_molecular_inputs(
    input_fnames: Path, data_root: Path, model_path: Path, output_dir: Path
    ) -> None:
    """Run inference on PDB and SDF files.

    Arguments:
        input_fnames: Space delimited file with two columns: paths to receptor 
            and ligand pdb and sdf files.
        data_root: Path to which files specified in input_fnames are relative.
        output_dir: Where to store the results.
    """
    output_parquets_dir = mkdir(output_dir / 'parquets')
    types_fname = output_dir / input_fnames.with_suffix('.types').name
    preds_fname = output_dir / 'predictions.txt'

    logging.basicConfig(filename=output_dir / str(int(time.time_ns() / 1e6)))
    logging.info('Generating types file...')
    generate_types_file(input_fnames, types_fname)
    logging.info('Done!')

    _, model, _, cmd_line_args, _ = get_model_and_test_dl(
            expand_path(model_path), types_fname, output_parquets_dir)

    lig_parser = StructuralFileParser(
        'ligand', cmd_line_args['extended_atom_types'])
    rec_parser = StructuralFileParser(
        'receptor', cmd_line_args['extended_atom_types'])

    lig_gts, rec_gts = [], []
    lig_sdfs, rec_pdbs = [], []
    with open(types_fname, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            rec, lig = line.strip().split()
            rec_gts.append(Path(output_parquets_dir, rec))
            lig_gts.append(Path(output_parquets_dir, lig))

    with open(input_fnames, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            rec, lig = line.strip().split()
            rec_pdbs.append(Path(data_root, rec))
            lig_sdfs.append(Path(data_root, lig))

    logging.info('Converting inputs to parquet format...')
    for lig_gt, lig_sdf in zip(lig_gts, lig_sdfs):
        lig_parser.file_to_parquets(
            lig_sdf, lig_gt.parent, lig_gt.name, False)

    for rec_gt, rec_pdb in zip(rec_gts, rec_pdbs):
        rec_parser.file_to_parquets(
            rec_pdb, rec_gt.parent, rec_gt.name, False)

    _, _, _, _, test_dl = get_model_and_test_dl(
        expand_path(model_path), types_fname, output_parquets_dir)
    logging.info('Done!')
    logging.info('Performing inference...\n')

    model.val(test_dl, preds_fname)
    with open(preds_fname, 'r', encoding='utf-8') as f:
        predictions = f.read().replace(' | ', ' ')
    with open(preds_fname, 'w', encoding='utf-8') as f:
        f.write(predictions)

    logging.info('\nDone!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fnames', '-i', required=True, type=str,
                        help='Space delimited file with two columns: paths to '
                             'receptor and ligand pdb and sdf files')
    parser.add_argument('--data_root', '-d', default='.', type=str,
                        help='Path relative to which files in --input_fnames '
                             'are specified (default is $pwd)')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Location of model directory, or saved pytorch '
                             'checkpoint file (*.pt)')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Where to store the output')
    args = parser.parse_args()

    input_fnames = expand_path(args.input_fnames)
    data_root = expand_path(args.data_root)
    output_dir = mkdir(args.output_dir)
    model_path = expand_path(args.model)

    predict_on_molecular_inputs(input_fnames, data_root, model_path, output_dir)
