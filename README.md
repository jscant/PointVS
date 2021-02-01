# PointVS: SE(3)-equivariant point cloud convolutions for virtual screening

## This project is in its nascent stages, so many features are not yet implemented and there are likely to be bugs.

[LieConv](https://github.com/mfinzi/LieConv) ([paper](https://arxiv.org/abs/2002.12880)) is a method for [group-equivariant](https://en.wikipedia.org/wiki/Equivariant_map) convolutions performed on point clouds in real space. [SE(3)-Transformer](https://github.com/FabianFuchsML/se3-transformer-public) ([paper](https://arxiv.org/abs/2006.10503)) is a similar method which also uses an attention-based graph neural network. Previous work on virtual screening using voxelisation can be found [here](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00263). PointVS uses LieConv and SE(3)-Transformer with SE(3) equivariance to perform [virtual screening](https://en.wikipedia.org/wiki/Virtual_screening).

No installation is required (yet). Dependencies are:

```
pytorch >= 1.7.1
LieConv
se3-transformer-public
wandb (optional but strongly recommended for easy graphs and logging)
```

A small working example is:

```
python3 lieconv_vs.py setrans data/small_chembl_test experiments/test_output
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
