from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


class SingleItemDataset(Dataset):

    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.graph


def get_pyg_single_graph_for_inference(graph):
    return list(DataLoader(SingleItemDataset(graph)))[0]
