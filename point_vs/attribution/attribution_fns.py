"""Perform masking on inputs"""
import numpy as np
import torch

from point_vs.models.point_neural_network import to_numpy


def cam(model, p, v, m, **kwargs):
    """Performs class activation mapping (CAM) on input.

    Arguments:
        p: matrix of size (1, n, 3) with atom positions
        v: matrix of size (1, n, d) with atom features
        m: matrix of ones of size (1, n)

    Returns:
        Numpy array containing CAM score attributions for each atom
    """
    x = model.group.lift((p, v, m), model.liftsamples)
    for layer in model.layers:
        if layer.__class__.__name__ == 'GlobalPool':
            break
        x = layer(x)
    final_layer_weights = to_numpy(model.layers[-1].weight).T
    node_features = to_numpy(x[1].squeeze())
    scores = node_features @ final_layer_weights
    return scores.squeeze()


def masking(model, p, v, m, bs=16):
    """Performs masking on each point in the input.

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
    p_input_matrix = torch.zeros(bs, p.size(1) - 1, p.size(2))
    v_input_matrix = torch.zeros(bs, v.size(1) - 1, v.size(2))
    m_input_matrix = torch.ones(bs, m.size(1) - 1).bool()
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
