from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.models.vanilla.lie_conv import LieResNet
from point_vs.models.vanilla.lie_transformer import EquivariantTransformer
from point_vs.preprocessing.data_loaders import PointCloudDataset, \
    PygPointCloudDataset
from .setup_and_params import MODEL_KWARGS, setup, DATALOADER_KWARGS

dump_path = setup()


def test_sartorras_egnn_instantiation():
    model = SartorrasEGNN(dump_path, 0, 0, None, None, **MODEL_KWARGS)
    assert model.save_path == dump_path


def test_lucid_egnn_instantiation():
    model = PygLucidEGNN(dump_path, 0, 0, None, None, **MODEL_KWARGS)
    assert model.save_path == dump_path


def test_lietransformer_instantiation():
    model = EquivariantTransformer(dump_path, 0, 0, None, None, **MODEL_KWARGS)
    assert model.save_path == dump_path


def test_lieconv_instantiation():
    model = LieResNet(dump_path, 0, 0, None, None, **MODEL_KWARGS)
    assert model.save_path == dump_path


def test_vanilla_data_loader_instantiation():
    test_dl = PygPointCloudDataset(dump_path, **DATALOADER_KWARGS)
    assert len(test_dl) == 1


def test_pyg_data_loader_instantiation():
    test_dl = PointCloudDataset(dump_path, **DATALOADER_KWARGS)
    assert len(test_dl) == 1
