"""Unit tests for the script written for Steph."""
import os
import tempfile

from pathlib import Path

from point_vs.scripts import for_steph


def test_generate_types_file():
    """Check types file is generated correctly."""
    output_types = tempfile.NamedTemporaryFile(  # pylint: disable=consider-using-with
        mode='w+', encoding='utf-8', delete=False)

    for_steph.generate_types_file(
        'test/resources/for_steph_test_input_files.txt',
        output_types.name)
    output_types.seek(0)
    with open(output_types.name, 'r', encoding='utf-8') as f:
        assert f.read() == 'resources/7zzp_rec_0.parquet resources/7zzp_lig_0.parquet\n'

    os.unlink(output_types.name)


def test_predict_on_molecular_inputs():
    """Check if all predictions are made."""
    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        for_steph.predict_on_molecular_inputs(
            input_fnames=Path(
                'test/resources/for_steph_test_input_files.txt'),
            data_root=Path('test'),
            model_path=Path(
                'test/resources/models/affinity_predictor'),
            output_dir=output_dir
        )
        with open(output_dir / 'predictions.txt', 'r', encoding='utf-8') as f:
            assert f.read() == '3.930 4.127 3.569 resources/7zzp_rec_0.parquet resources/7zzp_lig_0.parquet\n'
