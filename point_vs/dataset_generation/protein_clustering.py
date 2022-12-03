"""Protein clustering based on sequence similarity with cdhit."""
import argparse
from pathlib import Path

from point_vs import logging
from point_vs.dataset_generation.split_by_cdhit_output import \
    cdhit_output_to_graph
from point_vs.utils import expand_path, mkdir, execute_cmd


LOG = logging.get_logger('PointVS')


def filter_fasta_file(fasta_file, pdbids_file, output_file):
    with open(expand_path(pdbids_file), 'r') as f:
        pdbids = set([s.strip().lower() for s in f.readlines()])
    output = ''
    with open(expand_path(fasta_file), 'r') as f:
        buffer = ''
        for line in f.readlines():
            buffer += line.strip() + '\n'
            if line.startswith('>'):
                pdbid = line[1:5]
            else:
                if pdbid in pdbids:
                    output += buffer
                buffer = ''
    with open(expand_path(output_file), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', type=str, help='PDB info in FASTA format')
    parser.add_argument('test_pdbids', type=str, help='Test set PDBIDs file')
    parser.add_argument('train_pdbids', type=str, help='Train set PDBIDs file')
    parser.add_argument('output_dir', type=str, help='Where to save output')
    parser.add_argument('train_types', type=str, help='Original (baised) set '
                                                      'types file')
    parser.add_argument('--threshold', '-t', default=0.9,
                        help='Sequence similarity threshold')

    args = parser.parse_args()

    fasta = expand_path(args.fasta)
    test_pdbids = expand_path(args.test_pdbids)
    train_pdbids = expand_path(args.train_pdbids)
    output_dir = mkdir(args.output_dir)
    train_fasta = output_dir / 'train.fasta'
    test_fasta = output_dir / 'test.fasta'

    filter_fasta_file(fasta, train_pdbids, train_fasta)
    filter_fasta_file(fasta, test_pdbids, test_fasta)

    cmd_template = 'cd-hit-2d -i {0} -i2 {1} -o {2} -c {3} -M 80000 -b {4} -T ' \
                   '' \
                   '' \
                   '0 -n 5'
    execute_cmd(cmd_template.format(
        test_fasta,
        train_fasta,
        output_dir / 'cdhit_output',
        args.threshold,
        20
    ), silent=False, raise_exceptions=True)

    LOG.info('Constructing graph...')
    g = cdhit_output_to_graph(output_dir / 'cdhit_output.clstr')
    LOG.info('Done!')
    similar_pdbids = set()
    for key, value in g.items():
        similar_pdbids.add(key)
        similar_pdbids = similar_pdbids.union(set(value))
    similar_pdbids = list(similar_pdbids)
    LOG.info('Modifying types file...')
    new_types = ''
    with open(expand_path(args.train_types), 'r') as f:
        for line in f.readlines():
            add = True
            for pdbid in similar_pdbids:
                if pdbid in line.lower():
                    add = False
                    break
            if add:
                new_types += line
    with open(Path(output_dir,
                   Path(args.train_types).with_suffix('').name +
                   '_unbaised.types'), 'w') as f:
        f.write(new_types)
    LOG.info('Done!')
