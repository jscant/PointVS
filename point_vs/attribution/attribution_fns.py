"""Perform masking on inputs"""
import numpy as np
import torch
from torch_geometric.data import Data

from point_vs.models.geometric.pnn_geometric_base import PNNGeometricBase
from point_vs.models.point_neural_network_base import to_numpy
from point_vs.preprocessing.pyg_single_item_dataset import \
    get_pyg_single_graph_for_inference


def cam(model, p, v, m, edge_indices=None, edge_attrs=None, **kwargs):
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

        feats = model.get_embeddings(
            feats, edges, coords, edge_attributes, batch)
        x = to_numpy(model.layers[-1](feats))

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
            # the
            # dot product between each node's features and the final FC layer
            final_layer_weights = to_numpy(model.layers[-1].weight).T
            x = x @ final_layer_weights
            if liftsamples == 1:
                return x
            x = [np.mean(x[n:n + liftsamples]) for n in
                 range(len(x) // liftsamples)]
    return np.array(x)


def masking(model, p, v, m, bs=16):
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
    scores = np.zeros((m.size(1),))
    original_score = float(to_numpy(torch.sigmoid(model((p, v, m)))))
    p_input_matrix = torch.zeros(bs, p.size(1) - 1, p.size(2)).cuda()
    v_input_matrix = torch.zeros(bs, v.size(1) - 1, v.size(2)).cuda()
    m_input_matrix = torch.ones(bs, m.size(1) - 1).bool().cuda()
    for i in range(p.size(1) // bs):
        print(i * bs)
        for j in range(bs):
            global_idx = bs * i + j
            p_input_matrix[j, :, :] = p[0,
                                      torch.arange(p.size(1)) != global_idx, :]
            v_input_matrix[j, :, :] = v[0,
                                      torch.arange(v.size(1)) != global_idx, :]
        scores[i * bs:(i + 1) * bs] = to_numpy(torch.sigmoid(model((
            p_input_matrix, v_input_matrix,
            m_input_matrix)))).squeeze() - original_score
    for i in range(bs * (p.size(1) // bs), p.size(1)):
        masked_p = p[:, torch.arange(p.size(1)) != i, :].cuda()
        masked_v = v[:, torch.arange(v.size(1)) != i, :].cuda()
        masked_m = m[:, torch.arange(m.size(1)) != i].cuda()
        scores[i] = float(to_numpy(
            torch.sigmoid(
                model((masked_p, masked_v, masked_m))))) - original_score
    return scores
