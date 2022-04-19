import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

from point_vs.attribution.attribution import load_model
from point_vs.attribution.attribution_fns import atom_masking, cam
from point_vs.preprocessing.data_loaders import SynthPharmDataset
from point_vs.utils import expand_path, load_yaml, mkdir, PositionDict, \
    coords_to_string


def get_stats_from_dir(model_fname, directory, types, attribution_fn,
                       no_receptor=False):
    _, model, model_kwargs, cmd_line_args = load_model(model_fname)
    model = model.eval()
    directory = expand_path(directory)
    atom_labels_dict = load_yaml(directory.parent / 'atomic_labels.yaml')
    print('Loaded atomic labels')
    mol_label_dict = load_yaml(directory.parent / 'labels.yaml')
    print('Loaded molecular labels')
    lig_fnames, pharm_fnames, fname_indices = [], [], []
    lig_random_precisions = []
    rec_random_precisions = []
    lig_average_precisions = []
    rec_average_precisions = []
    rec_positions = []
    lig_positions = []

    ds = SynthPharmDataset(
        no_reeptor=no_receptor,
        base_path=directory,
        radius=cmd_line_args['radius'],
        polar_hydrogens=False,
        use_atomic_numbers=False,
        compact=True,
        types_fname=types,
        edge_radius=cmd_line_args['edge_radius'],
        estimate_bonds=cmd_line_args['estimate_bonds'],
        prune=cmd_line_args['prune']
    )
    dl = DataLoader(ds, 1, False)

    for graph in dl:
        lig_fname, rec_fname = graph.lig_fname[0], graph.rec_fname[0]

        fname_idx = int(Path(lig_fname.name).stem.split('lig')[-1])
        if not mol_label_dict[fname_idx]:
            continue

        pharm_fnames.append(rec_fname)
        lig_fnames.append(lig_fname)
        fname_indices.append(fname_idx)

        df = score_structure(model, graph, attribution_fn, no_receptor)
        df = label_df(df, PositionDict({
            coords_to_string(coord): True for coord in
            atom_labels_dict[fname_idx]}))

        df.sort_values(by=['attribution'], inplace=True, ascending=False)
        lig_df = df[df['bp'] == 0]
        rec_df = df[df['bp'] == 1]

        lig_positions += list(np.where(lig_df['y_true'] > 0.5)[0])[:1]
        rec_positions += list(np.where(rec_df['y_true'] > 0.5)[0])[:1]

        lig_random_pr = sum(lig_df['y_true']) / len(lig_df)
        rec_random_pr = sum(rec_df['y_true']) / len(rec_df)
        lig_pr = average_precision_score(
            lig_df['y_true'], lig_df['attribution'])
        rec_pr = average_precision_score(
            rec_df['y_true'], rec_df['attribution'])
        lig_random_precisions.append(lig_random_pr)
        rec_random_precisions.append(rec_random_pr)
        lig_average_precisions.append(lig_pr)
        rec_average_precisions.append(rec_pr)
    return (lig_random_precisions, lig_average_precisions,
            rec_random_precisions, rec_average_precisions,
            lig_positions, rec_positions)


def get_distances(lst):
    res = []
    for i in range(len(lst) // 2):
        res.append(
            np.linalg.norm(np.array(lst[2 * i]) - np.array(lst[2 * i + 1])))


def score_structure(model, graph, attribution_fn, no_receptor):
    h, edge_index, pos, edge_attr, batch = model.unpack_graph(graph)
    df = graph.df
    df = df[0]
    if no_receptor:
        df = df[df.bp == 0]
    attribution = attribution_fn(
        model, p=pos, v=h, edge_indices=edge_index, edge_attrs=edge_attr,
        synthpharm=True
    ).squeeze()
    df['attribution'] = attribution
    return df


def label_df(df, positions_list):
    labels = []
    coords_np = np.vstack([
        df['x'].to_numpy(),
        df['y'].to_numpy(),
        df['z'].to_numpy()
    ]).T
    for i in range(len(df)):
        coords = coords_np[i, :]
        labels.append(int(coords_to_string(coords) in positions_list))
    df['y_true'] = labels
    return df


def plot_rank_histogram(lig_ranks, rec_ranks, title=None, fname=None):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    max_rank = max(lig_ranks + rec_ranks)
    for idx, (ranks, subtitle) in enumerate(zip(
            [lig_ranks, rec_ranks], ['Ligand', 'Receptor'])):
        axs[idx].hist(ranks, density=False, bins=list(range(max_rank + 1)),
                      edgecolor='black', linewidth=1.0, color='blue')
        axs[idx].set_title(subtitle)
        axs[idx].set_xlabel('Top-scoring bonding atom rank')
        axs[idx].set_ylabel('Count')
    # fig.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if fname is not None:
        fname = expand_path(fname)
        mkdir(fname.parent)
        fig.savefig(fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Saved pytorch model weights')
    parser.add_argument('input_dir', type=str,
                        help='Location of ligand and receptor parquet files')
    parser.add_argument('types', type=str,
                        help='Types file containing locations of parquets '
                             'files relative to input_dir')
    parser.add_argument('output_dir', type=str, help='Where to store graphs')
    parser.add_argument('--no_receptor', '-n', action='store_true',
                        help='Do not include receptor information')
    args = parser.parse_args()

    for attr_fn, fn_name in zip((cam, atom_masking), ('CAM', 'Masking')):
        lrp, lap, rrp, rap, lig_positions, rec_positions = get_stats_from_dir(
            args.model, args.input_dir, args.types, attr_fn, args.no_receptor)
        if args.model.endswith('.pt'):
            project_name = expand_path(args.model).parents[1].name
        else:
            project_name = expand_path(args.model).name
        print()
        print('Project: {0}, Attribution method: {1}'.format(
            project_name, fn_name))
        print('Mean average precision (ligand):               {:.4f}'.format(
            np.mean(lap)))
        print('Random average precision (ligand):             {:.4f}'.format(
            np.mean(lrp)))
        print('Mean average precision (receptor):             {:.4f}'.format(
            np.mean(rap)))
        print('Random average precision (receptor):           {:.4f}'.format(
            np.mean(rrp)))
        print()
        print('Mean top scoring bonding atom rank (ligand):   {:.4f}'.format(
            np.mean(lig_positions)))
        print('Mean top scoring bonding atom rank (receptor): {:.4f}'.format(
            np.mean(rec_positions)))
        plot_rank_histogram(
            lig_positions, rec_positions, project_name,
            mkdir(args.output_dir) / 'rank_histogram_{0}_{1}.png'.format(
                fn_name, 'Project: {0}    Attibution method: {1}'.format(
                    project_name, fn_name)))
