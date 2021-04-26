"""Driver script for various graph attribution methods."""

import argparse
import urllib
from pathlib import Path

import torch
import yaml
from plip.basic.supplemental import extract_pdbid
from plip.exchange.webservices import fetch_pdb

from point_vs.attribution.attribution_fns import masking, cam
from point_vs.attribution.process_pdb import process_pdb
from point_vs.models.lie_conv import LieResNet
from point_vs.models.lie_transformer import EquivariantTransformer
from point_vs.utils import mkdir

ALLOWED_METHODS = ('masking', 'cam')


def download_pdb_file(pdbid, output_dir):
    """Given a PDB ID, downloads the corresponding PDB structure.
    Checks for validity of ID and handles error while downloading.
    Returns the path of the downloaded file (From PLIP)"""
    pdbid = pdbid.lower()
    output_dir = Path(output_dir).expanduser()
    pdbpath = output_dir / '{}.pdb'.format(pdbid)
    if pdbpath.is_file():
        print(pdbpath, 'already exists.')
        return pdbpath
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


def attribute(args):
    output_dir = mkdir(args.output_dir)
    pdbpath = download_pdb_file(args.pdbid, output_dir)

    model_path = Path(args.model).expanduser()
    with open(model_path.parents[1] / 'model_kwargs.yaml', 'r') as f:
        model_kwargs = yaml.load(f, Loader=yaml.Loader)
    with open(model_path.parents[1] / 'cmd_args.yaml', 'r') as f:
        cmd_line_args = yaml.load(f, Loader=yaml.Loader)

    model_type = cmd_line_args['model']
    dim_input = model_kwargs['dim_input']
    bs = cmd_line_args['batch_size']
    model_class = {
        'lietransformer': EquivariantTransformer,
        'lieconv': LieResNet
    }
    model_class = model_class[model_type]
    model = model_class(Path(), learning_rate=0.001, weight_decay=0,
                        silent=True, **model_kwargs)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    attribution_fn = {'masking': masking, 'cam': cam}[args.attribution_type]

    process_pdb(model, attribution_fn,
                str(pdbpath), str(output_dir), input_dim=dim_input,
                radius=cmd_line_args['radius'], bs=bs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attribution_type', type=str,
                        help='Method of graph attribution; just {} for '
                             'now.'.format(','.join(ALLOWED_METHODS)))
    parser.add_argument('model', type=str, help='Saved pytorch model')
    parser.add_argument('pdbid', type=str, help='PDB ID for structure to '
                                                'analyse')
    parser.add_argument('output_dir', type=str,
                        help='Directory in which to store results')
    args = parser.parse_args()
    attribute(args)
