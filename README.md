# PointVS: SE(3)-equivariant point cloud networks for virtual screening

## Introduction

[LieTransformer](https://github.com/oxcsml/lie-transformer)
([paper](https://arxiv.org/abs/2012.10885)) is a self-attention-based group 
equivariant network. [EGNN](https://github.com/vgsatorras/egnn)
([paper](https://arxiv.org/pdf/2102.09844.pdf)) is an SE(3)-equivariant
graph neural network. The author's previous work on deep learning for
virtual screening can be found 
[here](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00263). PointVS uses EGNN
and LieTransformer with SE(3) equivariance to perform
[virtual screening](https://en.wikipedia.org/wiki/Virtual_screening) and
pose selection.

##Installation and Initialisation
```
git clone https://github.com/jscant/PointVS
cd PointVS
conda env create -f environment.yml python=3.8
conda activate pointvs
pip install -e .
```
If you would like to use wandb for logging information on loss, performance and
hyperparameters (recommended), you must first create an account at
[wandb.ai](https://wandb.ai). You must then log into your account on your local
machine by opening a python3 console and executing:
```
import wandb
wandb.init()
```
and following the onscreen instructions.

A small working example is:
```
python3 point_vs.py egnn data/small_chembl_test test_output --test_types data/small_chembl_test.types
```

##Dataset generation

For full instructions, see the [README](https://github.com/jscant/PointVS/tree/master/point_vs/dataset_generation).

The input format for PointVS neural networks is the 
[parquet](http://parquet.apache.org/documentation/latest/). The script
`point_vs/dataset_generation/generate_types_file.py` can be used to generate
the types file (Loading Datasets section below), from which
`point_vs/dataset_generation/mol_to_parquet.py` can be used to convert 
pdb, sdf and mol2 files to the required format.

##Loading datasets
The recommended way to feed data into the models is to use types files, which
are used extensively in [GNINA 1.0](https://github.com/gnina/gnina). The
input `point_vs.py` arguments `train_data_root`, `--test_data_root (-t)`,
`--train_types` and `--test_types` should be specified.

###Option 1: Types File
Each line should follow the format:

`<label> <n/a> <rmsd> <receptor> <ligand>`

where:

`<label>` is a binary label (0 or 1)

`<n/a>` is any value (unused, there to keep in line with the original GNINA 
specification)

`<rmsd>` is the RMSD from the original crystal structure where the ligand is a 
docked structure and the RMSD is known, or -1 otherwise.

`<receptor>` is the relative location of the receptor structure in parquet format

`<ligand>` is the relative location of the ligand structure in parquet format

An example
types file can be found at `data/small_chembl_test.types`, where the `<rmsd>`
value is not known:

```
1 -1 -1.0 receptors/12968.parquet ligands/12968_actives/mol25_7.parquet
0 -1 -1.0 receptors/12968.parquet ligands/12968_decoys/mol9690_0.parquet
0 -1 -1.0 receptors/12968.parquet ligands/12968_decoys/mol6981_5.parquet
0 -1 -1.0 receptors/12968.parquet ligands/12968_decoys/mol2542_5.parquet
```

See the [README](https://github.com/jscant/PointVS/tree/master/point_vs/dataset_generation)
in `point_vs/dataset_generation` for a script which generates types files and 
calculates RMSDs.

###Option 2: Active and inactive molecules sorted by directory structure

The input structures should be in pandas-readable parquet files, with the ligand
and receptor in separate files for memory efficiency. The directory structure 
should be laid out as follows (the names of the folders `ligands` and 
`receptors` should be preserved, and directories containing ligand structures
should be named \<receptor\_name\>\_[actives|decoys], with receptor structures 
named \<receptor\_name\>.parquet):

```
dataset
├── ligands
│   ├── receptor_a_actives
│   │   ├──ligand_1.parquet
│   │   ├──ligand_2.parquet
│   ├── receptor_a_decoys
│   │   ├──ligand_3.parquet
│   │   ├──ligand_4.parquet
│   ├── receptor_b_actives
│   │   ├──ligand_5.parquet
│   │   ├──ligand_6.parquet
│   ├── receptor_b_decoys
│   │   ├──ligand_7.parquet
│   │   ├──ligand_8.parquet
└── receptors
    ├── receptor_a.parquet
    └── receptor_b.parquet
```

An example of the correct directory structure can be found in 
`data/small_chembl_test`.

