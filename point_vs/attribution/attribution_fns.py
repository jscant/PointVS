"""Various methods for attributing classifications decisions to inputs"""
import numpy as np
import torch
from scipy.stats import rankdata
from torch_geometric.data import Data

from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.models.geometric.pnn_geometric_base import PNNGeometricBase
from point_vs.models.point_neural_network_base import to_numpy
from point_vs.preprocessing.pyg_single_item_dataset import \
    get_pyg_single_graph_for_inference
from point_vs.utils import expand_path

SIGMOID = False

def attention_wrapper(**kwargs):
    """Dummy fn."""
    pass


def cam_wrapper(**kwargs):
    """Dummy fn."""
    pass


def masking_wrapper(**kwargs):
    """Dummy_fn."""


def bond_masking(model, p, v, m=None, bs=32, edge_indices=None, edge_attrs=None,
                 **kwargs):

    def extract_score(graph):
        if SIGMOID:
            original_score = to_numpy(torch.sigmoid(model(graph)).squeeze())
        else:
            original_score = to_numpy(model(graph)).squeeze()
        if len(original_score) > 1:
            return float(original_score[1])
        return float(original_score)

    global SIGMOID
    graph = get_pyg_single_graph_for_inference(Data(
        x=v.squeeze(),
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        pos=p.squeeze(),
    ))

    scores = []
    original_score = extract_score(graph)

    for i in range(edge_indices.shape[1]):

        if not edge_attrs[i, 1]:
            scores.append(0)
            continue

        atom_a = edge_indices[0, i]
        atom_b = edge_indices[1, i]
        atom_1 = min(atom_a, atom_b)
        atom_2 = max(atom_a, atom_b)

        p_input_matrix = torch.cat([
            p[:, :atom_1, :],
            p[:, atom_1 + 1:atom_2, :],
            p[:, atom_2 + 1:, :],
        ], dim=1)
        v_input_matrix = torch.cat([
            v[:, :atom_1, :],
            v[:, atom_1 + 1:atom_2, :],
            v[:, atom_2 + 1:, :],
        ], dim=1)

        edge_minus_idx_prod_1 = torch.prod(edge_indices - atom_1, dim=0)
        edge_minus_idx_prod_2 = torch.prod(edge_indices - atom_2, dim=0)

        edge_minus_idx_prod = torch.from_numpy(to_numpy(
            edge_minus_idx_prod_1 * to_numpy(edge_minus_idx_prod_2)))

        # At this point we have 0s where we want to mask the edge_indices
        # = [0, 0, 0, 1, 10, 30, ...], shape = len(edge_indices)
        mask = edge_minus_idx_prod != 0

        edge_indices_copy = torch.clone(edge_indices)
        edge_indices_copy[torch.where(edge_indices > atom_1)] -= 1
        edge_indices_copy[torch.where(edge_indices > atom_2)] -= 1

        masked_edge_attributes = edge_attrs[mask, :]
        masked_edge_indices = edge_indices_copy[:, mask]

        graph = get_pyg_single_graph_for_inference(Data(
            x=v_input_matrix.squeeze(),
            edge_index=masked_edge_indices,
            edge_attr=masked_edge_attributes,
            pos=p_input_matrix.squeeze(),
        ))

        scores.append(
            original_score - extract_score(graph))
        if not i % 500:
            print('{0}/{1}'.format(i, edge_indices.shape[1]), scores[-1],
                  max(scores))
    return np.array(scores)


def track_bond_lengths(
        model, p, v, edge_indices=None, edge_attrs=None, **kwargs):
    graph = get_pyg_single_graph_for_inference(Data(
        x=v.squeeze(),
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        pos=p.squeeze(),
    ))

    # put the graph through the model before extracting stored weights
    model(graph)
    original_coords = model.layers[0].intermediate_coords
    final_coords = model.layers[-1].intermediate_coords
    all_bond_lengths = []
    for coords in [original_coords, final_coords]:
        bond_lengths = []
        for edge_idx in range(edge_indices.shape[1]):
            i, j = edge_indices[0, edge_idx], edge_indices[1, edge_idx]
            bond_lengths.append(np.linalg.norm(coords[i, :] - coords[j, :]))
        all_bond_lengths.append(np.array(bond_lengths))
    bond_change = all_bond_lengths[1] - all_bond_lengths[0]
    return bond_change


def track_position_changes(
        model, p, v, edge_indices=None, edge_attrs=None, **kwargs):
    graph = get_pyg_single_graph_for_inference(Data(
        x=v.squeeze(),
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        pos=p.squeeze(),
    ))

    # put the graph through the model before extracting stored weights
    model(graph)
    original_coords = model.layers[0].intermediate_coords
    displacements = []
    for layer in range(1, model.n_layers + 1):
        displacement = model.layers[layer].intermediate_coords - original_coords
        sq_displacement = np.sum(displacement ** 2, axis=1)
        displacements.append(np.sqrt(sq_displacement))
        # original_coords = model.layers[layer].intermediate_coords
    displacements = np.vstack(displacements).T
    print(np.sum(displacements, axis=1))
    return np.sum(displacements, axis=1)


def extract_coords_from_pdb_line(line, as_str=True):
    """Obtain coordinates from a line in a PDB file."""
    line = line.strip()
    x = line[30:38].strip()
    y = line[38:46].strip()
    z = line[46:54].strip()
    if not as_str:
        return float(x), float(y), float(z)
    return x, y, z


def replace_coords_line(line, x, y, z):
    """Replace the coordinates from a line in a PDB file with new x, y, z."""
    x, y, z = str(x)[:8], str(y)[:8], str(z)[:8]
    x = ' ' * (8 - len(x)) + x
    y = ' ' * (8 - len(y)) + y
    z = ' ' * (8 - len(z)) + z
    new_line = line[:30] + x + y + z + line[54:]
    return new_line


def replace_coords(input_pdb, output_pdb, old_coords, new_coords):
    """Replace coordinates of all atoms in a PDB file.

    Arguments:
        input_pdb: original PDB file with original coordinates
        output_pdb: name of output PDB file
        old_coords: pd.DataFrame containing old coordinates to be replaced
        new_coords: pd.DataFrame containing new coordinates
    """
    input_pdb = expand_path(input_pdb)
    old_to_new_map = {}
    for idx in range(old_coords.shape[0]):
        old_as_str = ':'.join([str(x) for x in old_coords[idx, :]])
        new_as_str = ':'.join([str(x) for x in new_coords[idx, :]])
        old_to_new_map[old_as_str] = new_as_str
    new_pdb = ''
    with open(input_pdb, 'r') as f:
        for line in f.readlines():
            if not line.startswith('ATOM') and not line.startswith('HETATM'):
                new_pdb += line
                continue
            coords_str = ':'.join(extract_coords_from_pdb_line(line))
            try:
                new_coords = old_to_new_map[coords_str]
            except KeyError:
                new_pdb += line
            else:
                new_pdb += replace_coords_line(line, *new_coords.split(':'))
    with open(expand_path(output_pdb), 'w') as f:
        f.write(new_pdb)


def mean_node_attention_rank(
        model, p, v, edge_indices=None, edge_attrs=None, gnn_layer=-1,
        **kwargs):
    graph = get_pyg_single_graph_for_inference(Data(
        x=v.squeeze(),
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        pos=p.squeeze(),
    ))

    # put the graph through the model before extracting stored weights
    model(graph)
    ranks = []
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'node_att_val') and layer.node_att_val is not None:
            if idx == 10:
                break
            att_scores = layer.node_att_val
            ranks.append(rankdata(att_scores.flatten()) - 1)
    return np.mean(np.vstack(ranks).T, axis=1)


def mean_edge_attention_rank(
        model, p, v, edge_indices=None, edge_attrs=None, gnn_layer=-1,
        **kwargs):
    assert isinstance(model, PNNGeometricBase), \
        'Attention based attribution only compatable with SartorrasEGNN'
    graph = get_pyg_single_graph_for_inference(Data(
        x=v.squeeze(),
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        pos=p.squeeze(),
    ))

    # put the graph through the model before extracting stored weights
    model(graph)
    ranks = []
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, 'att_val') and layer.att_val is not None:
            if idx == 10:
                break
            att_scores = layer.att_val
            ranks.append(rankdata(att_scores.flatten()) - 1)
            # ranks.append(att_scores.flatten())
    return np.mean(np.vstack(ranks).T, axis=1)


def node_attention(
        model, p, v, edge_indices=None, edge_attrs=None, gnn_layer=-1,
        **kwargs):
    """Use node attention weights to assign importance to input atoms.

    Arguments:
        model: trained GNN
        p: matrix of size (1, n, 3) with atom positions
        v: matrix of size (1, n, d) with atom features
        edge_indices: (EGNN) indices of connected atoms
        edge_attrs: (EGNN) type of bond (inter/intra ligand/receptor)
        gnn_layer:

    Returns:
        Numpy array containing attention weight scores attributions for each
        atom.
    """
    global SIGMOID

    def inverse_sigmoid(x):
        return np.log(x / (1 - x))

    graph = get_pyg_single_graph_for_inference(Data(
        x=v.squeeze(),
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        pos=p.squeeze(),
    ))

    # put the graph through the model before extracting stored weights
    model(graph)
    if not SIGMOID:
        return model.layers[gnn_layer].node_att_val.reshape((-1,))
    return inverse_sigmoid(model.layers[gnn_layer].node_att_val.reshape((-1,)))


def edge_attention(
        model, p, v, edge_indices=None, edge_attrs=None, gnn_layer=-1,
        **kwargs):
    assert isinstance(model, PNNGeometricBase), \
        'Attention based attribution only compatable with SartorrasEGNN'
    graph = get_pyg_single_graph_for_inference(Data(
        x=v.squeeze(),
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        pos=p.squeeze(),
    ))

    # put the graph through the model before extracting stored weights
    model(graph)
    return model.layers[gnn_layer].att_val.reshape((-1,))


def edge_embedding_attribution(
        model, p, v, edge_indices=None, edge_attrs=None, **kwargs):
    assert isinstance(model, SartorrasEGNN), \
        'Edge based attribution only compatable with SartorrasEGNN'
    graph = get_pyg_single_graph_for_inference(Data(
        x=v.squeeze(),
        edge_index=edge_indices,
        edge_attr=edge_attrs,
        pos=p.squeeze(),
    ))

    feats, edges, coords, edge_attributes, batch = model.unpack_graph(
        graph)
    _, edge_embeddings = model.get_embeddings(
        feats, edges, coords, edge_attributes, batch)
    edge_scores = to_numpy(model.edges_linear_layers(edge_embeddings))

    return edge_scores


def cam(model, p, v, m=None, edge_indices=None, edge_attrs=None, **kwargs):
    """Perform class activation mapping (CAM) on input.

    Arguments:
        p: matrix of size (1, n, 3) with atom positions
        v: matrix of size (1, n, d) with atom features
        m: matrix of ones of size (1, n)
        edge_indices: (EGNN) indices of connected atoms
        edge_attrs: (EGNN) type of bond (inter/intra ligand/receptor)

    Returns:
        Numpy array containing CAM score attributions for each atom
    """
    if isinstance(model, PNNGeometricBase):
        graph = get_pyg_single_graph_for_inference(Data(
            x=v.squeeze(),
            edge_index=edge_indices,
            edge_attr=edge_attrs,
            pos=p.squeeze(),
        ))
        feats, edges, coords, edge_attributes, batch = model.unpack_graph(
            graph)

        feats, _ = model.get_embeddings(
            feats, edges, coords, edge_attributes, batch)
        x = to_numpy(model.feats_linear_layers(feats))
        if len(x.shape) == 2 and x.shape[1] == 3:
            x = np.mean(x, axis=1)

    else:
        if hasattr(model, 'group') and hasattr(model.group, 'lift'):
            x = model.group.lift((p, v, m), model.liftsamples)
            liftsamples = model.liftsamples
        else:
            x = p, v, m
            liftsamples = 1
        for layer in model.layers:
            if layer.__class__.__name__.find('GlobalPool') != -1:
                break
            x = layer(x)
        x = to_numpy(x[1].squeeze())
        if not model.linear_gap:
            # We can directly look at the contribution of each node by taking
            # the dot product between each node's features and the final FC
            # layer
            final_layer_weights = to_numpy(model.layers[-1].weight).T
            x = x @ final_layer_weights
            if liftsamples == 1:
                return x
            x = [np.mean(x[n:n + liftsamples]) for n in
                 range(len(x) // liftsamples)]
    return np.array(x)


def atom_masking(
        model, p, v, m=None, bs=32, edge_indices=None, edge_attrs=None,
        resis=None, **kwargs):
    """Perform masking on each point in the input.

    Scores are calculated by taking the difference between the original
    (unmasked) score and the score with each point masked.

    Arguments:
        p: matrix of size (1, n, 3) with atom positions
        v: matrix of size (1, n, d) with atom features
        m: matrix of ones of size (1, n)
        bs: batch size to use (larger is faster but requires more GPU memory)

    Returns:
        Numpy array containing masking score attributions for each atom
    """
    global SIGMOID
    # Number of atoms

    if kwargs.get('synthpharm', False):
        n_atoms = p.size(0)
    else:
        n_atoms = p.size(1)
    scores = np.zeros((n_atoms,))

    if isinstance(model, PNNGeometricBase):
        graph = get_pyg_single_graph_for_inference(Data(
            x=v.squeeze(),
            edge_index=edge_indices,
            edge_attr=edge_attrs,
            pos=p.squeeze(),
        ))

        original_score = model(graph)
        is_regression = len(
            original_score.shape) == 2 and original_score.shape[1] == 3
        if is_regression:
            original_score = np.mean(to_numpy(original_score.squeeze()))
        else:
            if SIGMOID:
                original_score = float(to_numpy(torch.sigmoid(original_score)))
            else:
                original_score = float(to_numpy(original_score))
        if kwargs.get('synthpharm', False):
            p = p.reshape(1, *p.shape)
            v = v.reshape(1, *v.shape)
        for i in range(n_atoms):
            p_input_matrix = torch.zeros(p.size(1) - 1, p.size(2)).cuda()
            v_input_matrix = torch.zeros(v.size(1) - 1, v.size(2)).cuda()

            p_input_matrix[:i, :] = p[:, :i, :]
            p_input_matrix[i:, :] = p[:, i + 1:, :]
            v_input_matrix[:i, :] = v[:, :i, :]
            v_input_matrix[i:, :] = v[:, i + 1:, :]

            edge_minus_idx_prod = torch.prod(edge_indices - i, dim=0)
            mask = torch.where(edge_minus_idx_prod)
            e_attrs_input_matrix = edge_attrs[mask]
            e_indices_input_matrix = edge_indices.T[mask].T

            e_indices_input_matrix[np.where(
                e_indices_input_matrix.cpu() > i)] -= 1

            graph = get_pyg_single_graph_for_inference(Data(
                x=v_input_matrix.squeeze(),
                edge_index=e_indices_input_matrix,
                edge_attr=e_attrs_input_matrix,
                pos=p_input_matrix.squeeze(),
            ))
            x = model(graph)
            if is_regression:
                scores[i] = original_score - np.mean(to_numpy(x.squeeze()))
            else:
                if SIGMOID:
                    scores[i] = original_score - float(to_numpy(torch.sigmoid(x)))
                else:
                    scores[i] = original_score - float(to_numpy(x))
    else:
        original_score = float(to_numpy(torch.sigmoid(model((p, v, m)))))
        p_input_matrix = torch.zeros(bs, p.size(1) - 1, p.size(2)).cuda()
        v_input_matrix = torch.zeros(bs, v.size(1) - 1, v.size(2)).cuda()
        m_input_matrix = torch.ones(bs, m.size(1) - 1).bool().cuda()
        for i in range(p.size(1) // bs):
            print(i * bs)
            for j in range(bs):
                global_idx = bs * i + j
                p_input_matrix[j, :, :] = p[0,
                                          torch.arange(p.size(1)) != global_idx,
                                          :]
                v_input_matrix[j, :, :] = v[0,
                                          torch.arange(v.size(1)) != global_idx,
                                          :]
            scores[i * bs:(i + 1) * bs] = to_numpy(torch.sigmoid(model((
                p_input_matrix, v_input_matrix,
                m_input_matrix)))).squeeze() - original_score
        for i in range(bs * (p.size(1) // bs), p.size(1)):
            masked_p = p[:, torch.arange(p.size(1)) != i, :].cuda()
            masked_v = v[:, torch.arange(v.size(1)) != i, :].cuda()
            masked_m = m[:, torch.arange(m.size(1)) != i].cuda()
            scores[i] = original_score - float(to_numpy(torch.sigmoid(
                model((masked_p, masked_v, masked_m)))))
    return scores