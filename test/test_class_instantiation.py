"""Check if important classes are correctly instantiated."""
from point_vs.models.geometric.egnn_lucid import PygLucidEGNN
from point_vs.models.geometric.egnn_satorras import SartorrasEGNN
from point_vs.preprocessing.data_loaders import PointCloudDataset, \
    PygPointCloudDataset
from .setup_and_params import MODEL_KWARGS, setup, DATALOADER_KWARGS

dump_path = setup()


def test_sartorras_egnn_instantiation():
    """Check if satorras-egnn is instantiated."""
    model = SartorrasEGNN(dump_path, 0, 0, None, None, **MODEL_KWARGS)
    assert model.save_path == dump_path


def test_lucid_egnn_instantiation():
    """Check if lucid-egnn is instantiated."""
    model = PygLucidEGNN(dump_path, 0, 0, None, None, **MODEL_KWARGS)
    assert model.save_path == dump_path


def test_vanilla_data_loader_instantiation():
    """Check if data_loader is instantiated."""
    test_dl = PygPointCloudDataset(dump_path, **DATALOADER_KWARGS)
    assert len(test_dl) == 1


def test_pyg_data_loader_instantiation():
    """Check is pyg data_loader is instantiated."""
    test_dl = PointCloudDataset(dump_path, **DATALOADER_KWARGS)
    assert len(test_dl) == 1
