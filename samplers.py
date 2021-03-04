import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, Sampler


class SubsetSequentialSampler(SubsetRandomSampler):

    def __iter__(self):
        return (self.indices[i] for i in np.arange(len(self.indices)))


class WeightedSubsetRandomSampler(Sampler):
    """WeightedRandomSampler but with a subset."""

    def __init__(self, weights, indices, replacement=True, generator=None):
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.np_indices = np.array(indices)
        self.np_subweights = np.array(weights)[self.np_indices]

        self.subweights = torch.as_tensor(self.np_subweights)

        self.replacement = replacement
        self.generator = generator
        self.num_samples = len(self.np_indices)

    def __iter__(self):
        rand_tensor = torch.multinomial(
            self.subweights, len(self.np_subweights), self.replacement,
            generator=self.generator).numpy()
        return iter(list(self.np_indices[rand_tensor]))

    def __len__(self):
        return len(self.np_indices)