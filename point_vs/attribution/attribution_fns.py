"""Perform masking on inputs"""
import numpy as np
import torch

from point_vs.models.point_neural_network import to_numpy
from point_vs.utils import Timer


def masking(model, p, v, m, bs=16):
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
    for i in range(p.size(1)):
        with Timer() as t:
            masked_p = p[:, torch.arange(p.size(1)) != i, :].cuda()
            masked_v = v[:, torch.arange(v.size(1)) != i, :].cuda()
            masked_m = m[:, torch.arange(m.size(1)) != i].cuda()
        with Timer() as f:
            scores[i] = float(to_numpy(
                torch.sigmoid(
                    model((masked_p, masked_v, masked_m))))) - original_score
        print(i, 'Making tensors:', t.interval, '\tScoring:', f.interval)
    return scores
