# LieConvVS: SE(3)-equivariant point cloud convolutions for virtual screening

## This project is in its nascent stages, so many features are not yet implemented and there are likely to be bugs.

[LieConv](https://github.com/mfinzi/LieConv) ([paper](https://arxiv.org/abs/2002.12880)) is a method for [group-equivariant](https://en.wikipedia.org/wiki/Equivariant_map) convolutions performed on point clouds in real space. LieConvVS uses LieConv with SE(3) equivariance to perform [virtual screening](https://en.wikipedia.org/wiki/Virtual_screening). Previous work on virtual screening using voxelisation can be found [here](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00263).

No installation is required (yet). Dependencies are:

```
pytorch >= 1.7.1
LieConv
```

A small working example is:

```
python3 lieconv_vs.py resnet data/small_chembl_test ~/test_output
```

