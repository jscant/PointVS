# PointVS: SE(3)-equivariant point cloud networks for virtual screening

## This project is in its nascent stages, so many features are not yet implemented and there are likely to be bugs.

[LieTransformer](https://github.com/oxcsml/lie-transformer)
([paper](https://arxiv.org/abs/2012.10885)) is a self-attention-based group 
equivariant network.. [LieConv](https://github.com/mfinzi/LieConv)
([paper](https://arxiv.org/abs/2002.12880)) is a method for
[group-equivariant](https://en.wikipedia.org/wiki/Equivariant_map) convolutions
performed on point clouds in real space. Previous work on virtual screening
using voxelisation can be found
[here](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00263). PointVS uses LieConv
and LieTransformer with SE(3) equivariance to perform
[virtual screening](https://en.wikipedia.org/wiki/Virtual_screening).

A conda installation is provided:
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
python3 point_vs.py egnn data/small_chembl_test experiments/test_output
```

The input structures should be in pandas-readable parquet files, with the ligand and receptor in separate files for memory efficiency. The directory structure should be laid out as follows (the names of the folders `ligands` and `receptors` should be preserved, and directories containing ligand structures should be named \<receptor\_name\>\_[actives|decoys], with receptor structures named \<receptor\_name\>.parquet):

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
