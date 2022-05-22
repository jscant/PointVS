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

## Installation and Initialisation
You are required to have SSH keys set up between your machine and GitHub
for setup of some external pip dependencies.
```
git clone https://github.com/jscant/PointVS
cd PointVS
git remote set-url origin git@github.com:jscant/PointVS.git
conda env create -f environment.yml python=3.8
conda activate pointvs
pip install -e .
```

To run tests for invariance under SE(3) transformations:
```
python3 -m pytest -vvv
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
python3 point_vs.py egnn data/small_chembl_test test_output --train_types data/small_chembl_test.types --egnn_classify_on_feats
```

## Usage

```
Usage: point_vs.py model train_data_root save_path

positional arguments:
  model                 Type of point cloud network to use: lietransformer, lieconv, lucid or egnn
  train_data_root       Location of structure training *.parquets files. Receptors should be in a
                        directory named receptors, with ligands located in their specific receptor
                        subdirectory under the ligands directory.
  save_path             Directory in which experiment outputs are stored. If wandb_run and
                        wandb_project are specified, save_path/wandb_project/wandb_run will be used
                        to store results.

optional arguments:
  -h, --help            show this help message and exit
  --load_weights LOAD_WEIGHTS, -l LOAD_WEIGHTS
                        Load a model.
  --test_data_root TEST_DATA_ROOT, -t TEST_DATA_ROOT
                        Location of structure test *.parquets files. Receptors should be in a
                        directory named receptors, with ligands located in their specific receptor
                        subdirectory under the ligands directory.
  --translated_actives TRANSLATED_ACTIVES
                        Directory in which translated actives are stored. If unspecified, no
                        translated actives will be used. The use of translated actives are is
                        discussed in https://pubs.acs.org/doi/10.1021/acs.jcim.0c00263
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Number of examples to include in each batch for training.
  --epochs EPOCHS, -e EPOCHS
                        Number of times to iterate through training set.
  --channels CHANNELS, -k CHANNELS
                        Channels for feature vectors
  --train_receptors [TRAIN_RECEPTORS [TRAIN_RECEPTORS ...]], -r [TRAIN_RECEPTORS [TRAIN_RECEPTORS ...]]
                        Names of specific receptors for training. If specified, other structures
                        will be ignored.
  --test_receptors [TEST_RECEPTORS [TEST_RECEPTORS ...]], -q [TEST_RECEPTORS [TEST_RECEPTORS ...]]
                        Names of specific receptors for testing. If specified, other structures
                        will be ignored.
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate for gradient descent
  --weight_decay WEIGHT_DECAY, -w WEIGHT_DECAY
                        Weight decay for regularisation
  --wandb_project WANDB_PROJECT
                        Name of wandb project. If left blank, wandb logging will not be used.
  --wandb_run WANDB_RUN
                        Name of run for wandb logging.
  --layers LAYERS       Number of layers in LieResNet
  --liftsamples LIFTSAMPLES
                        liftsamples parameter in LieConv
  --radius RADIUS       Maximum distance from a ligand atom for a receptor atom to be included in
                        input
  --nbhd NBHD           Number of monte carlo samples for integral
  --load_args LOAD_ARGS
                        Load yaml file with command line args. Any args specified in the file will
                        overwrite other args specified on the command line.
  --double              Use 64-bit floating point precision
  --kernel_type KERNEL_TYPE
                        One of 2232, mlp, overrides attention_fn (see original repo)
                        (LieTransformer)
  --attention_fn ATTENTION_FN
                        One of norm_exp, softmax, dot_product: activation for attention (overridden
                        by kernel_type) (LieTransformer)
  --activation ACTIVATION
                        Activation function
  --kernel_dim KERNEL_DIM
                        Size of linear layers in attention kernel (LieTransformer)
  --feature_embed_dim FEATURE_EMBED_DIM
                        Feature embedding dimension for attention; paper had dv=848 for QM9
                        (LieTransformer)
  --mc_samples MC_SAMPLES
                        Monte carlo samples for attention (LieTransformer)
  --dropout DROPOUT     Chance for nodes to be inactivated on each trainin batch (EGNN)
  --fill FILL           LieTransformer fill parameter
  --use_1cycle          Use 1cycle learning rate scheduling
  --warm_restarts       Use cosine annealing with warm restarts
  --fourier_features FOURIER_FEATURES
                        (EGNN) Number of fourier terms to use when encoding distances (default is
                        not to use fourier distance encoding)
  --norm_coords         (EGNN) Normalise coordinate vectors
  --norm_feats          (EGNN) Normalise feature vectors
  --use_atomic_numbers  Use atomic numbers rather than smina types
  --compact             Use compact rather than true one-hot encodings
  --thin_mlps           (EGNN) Use single layer MLPs for edge, node and coord updates
  --hydrogens           Include polar hydrogens
  --augmented_actives AUGMENTED_ACTIVES
                        Number of randomly rotated actives to be included as decoys during training
  --min_aug_angle MIN_AUG_ANGLE
                        Minimum angle of rotation for augmented actives as specified in the
                        augmented_actives argument
  --max_active_rmsd MAX_ACTIVE_RMSD
                        (Pose selection) maximum non-aligned RMSD between the original crystal pose
                        and active redocked poses
  --min_inactive_rmsd MIN_INACTIVE_RMSD
                        (Pose selection) minimum non-aligned RMSD between original crystal pose and
                        inactive redocked poses
  --val_on_epoch_end, -v
                        Run inference ion the validation set at the end of every epoch during
                        training
  --synth_pharm, -p     Synthetic Pharmacophore mode (for Tom, beta)
  --input_suffix INPUT_SUFFIX, -s INPUT_SUFFIX
                        Filename extension for inputs
  --train_types TRAIN_TYPES
                        Optional name of GNINA-like types file which contains paths and labels for
                        a training set. See GNINA 1.0 documentation for specification.
  --test_types TEST_TYPES
                        Optional name of GNINA-like types file which contains paths and labels for
                        a test set. See GNINA 1.0 documentation for specification.
  --egnn_attention      Use attention mechanism on edges for EGNN
  --egnn_tanh           Put tanh layer at the end of the coordinates mlp (EGNN)
  --egnn_normalise      Normalise radial coordinates (EGNN)
  --egnn_residual       Use residual connections (EGNN)
  --edge_radius EDGE_RADIUS
                        Maximum interatomic distance for an edge to exist (EGNN)
  --end_flag            Add a file named "_FINISHED" to the save_path upon training and test
                        completion
  --wandb_dir WANDB_DIR
                        Location to store wandb files. Defaults to
                        <save_path>/<wandb_project>/<wandb_run>/wandb.
  --estimate_bonds      (EGNN): Instead of using a fixed edge radius,the intermolecular radius is
                        set at --edge_radius Angstroms but the intramolecular radius is set at 2A,
                        which has the effect of putting edges where there are covalent bonds
                        between atoms in the same molecule.
  --linear_gap          Final linear layer comes after rather than before the global average
                        pooling layer. This can improve performance significantly.
  --prune               (EGNN) Prune subgraphs which are not connected to the ligand
  --top1                A poorly kept secret ;)
```

## Dataset generation

For full instructions, see the [README](https://github.com/jscant/PointVS/tree/master/point_vs/dataset_generation).

The input format for PointVS neural networks is the 
[parquet](http://parquet.apache.org/documentation/latest/). The script
`point_vs/dataset_generation/generate_types_file.py` can be used to generate
the types file (Loading Datasets section below), from which
`point_vs/dataset_generation/types_to_parquet.py` can be used to convert 
pdb, sdf and mol2 files to the required format.

## Loading datasets
The recommended way to feed data into the models is to use types files, which
are used extensively in [GNINA 1.0](https://github.com/gnina/gnina). The
input `point_vs.py` arguments `train_data_root`, `--test_data_root (-t)`,
`--train_types` and `--test_types` should be specified.

### Option 1: Types File
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

### Option 2: Active and inactive molecules sorted by directory structure

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

