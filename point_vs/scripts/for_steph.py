"""PDB/sdf combinations -> score"""

import argparse
from pathlib import Path

from point_vs.dataset_generation.types_to_parquet import \
    StructuralFileParser
from point_vs.inference import get_model_and_test_dl
from point_vs.utils import expand_path, mkdir


def generate_types_file(input_fnames, types_fname):
    """Generate a types file from the inputs, replacing extensions."""
    types = ''
    line_template = '{0} {1}\n'
    with open(input_fnames, 'r') as f:
        for line in f.readlines():
            chunks = line.split()
            if len(chunks) != 2:
                continue
            rec_pdb, lig_sdf = chunks
            rec_gt = rec_pdb.replace('.pdb', '.parquet')
            lig_gt = lig_sdf.replace(
                '.sdf', '.mol2').replace('.mol2', '.parquet')
            types += line_template.format(rec_gt, lig_gt)
    with open(expand_path(types_fname), 'w') as f:
        f.write(types)


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
    output_parquets_dir = mkdir(output_dir / 'parquets')
    types_fname = Path(output_dir, input_fnames.with_suffix('').name + '.types')
    preds_fname = output_dir / 'predictions.txt'

    print('Generating types file...')
    generate_types_file(input_fnames, types_fname)
    print('Done!')

    checkpoint_path, model, model_kwargs, cmd_line_args, test_dl = \
        get_model_and_test_dl(
            expand_path(args.model), types_fname, output_parquets_dir)

    regression = cmd_line_args['model_task'] != 'classification'

    lig_parser = StructuralFileParser(
        'ligand', cmd_line_args['extended_atom_types'])
    rec_parser = StructuralFileParser(
        'receptor', cmd_line_args['extended_atom_types'])

    lig_gts, rec_gts = [], []
    lig_sdfs, rec_pdbs = [], []
    with open(types_fname, 'r') as f:
        for line in f.readlines():
            rec, lig = line.strip().split()
            rec_gts.append(Path(output_parquets_dir, rec))
            lig_gts.append(Path(output_parquets_dir, lig))

    with open(input_fnames, 'r') as f:
        for line in f.readlines():
            rec, lig = line.strip().split()
            rec_pdbs.append(Path(data_root, rec))
            lig_sdfs.append(Path(data_root, lig))

    print('Converting inputs to parquet format...')
    for lig_gt, lig_sdf in zip(lig_gts, lig_sdfs):
        lig_parser.file_to_parquets(
            lig_sdf, lig_gt.parent, lig_gt.name, False)

    for rec_gt, rec_pdb in zip(rec_gts, rec_pdbs):
        rec_parser.file_to_parquets(
            rec_pdb, rec_gt.parent, rec_gt.name, False)
    print('Done!')
    print('Performing inference...\n')

    model.val(test_dl, preds_fname)
    with open(preds_fname, 'r') as f:
        predictions = f.read().replace(' | ', ' ')
    with open(preds_fname, 'w') as f:
        f.write(predictions)

    print('\nDone!')
