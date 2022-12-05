# PointVS: SE(3)-equivariant point cloud networks for virtual screening

## Introduction

[EGNN](https://github.com/vgsatorras/egnn)
([paper](https://arxiv.org/pdf/2102.09844.pdf)) is an E(3)-equivariant
graph neural network layer. In this project, we use networks based on this to
make E(3)-invariant predictions of binding pose and affinity.

The author's previous work on deep learning for
virtual screening can be found 
[here](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00263).

## Installation (Linux/MacOS Bash, MacOS zsh):
Copy and paste the following commands:
```
ARCH=$(arch)
if [[ "$OSTYPE" == "linux-gnu"* && "$ARCH" == "x86_64"* ]]; then
    INSTALLSCRIPT="environment.yml"
elif [[ "$OSTYPE" == "darwin"* && "$ARCH" == "arm"* ]]; then
    INSTALLSCRIPT="environment_apple_silicon.yml"
elif [[ "$OSTYPE" == "darwin"* && "$ARCH" == "x86_64"* ]]; then
    INSTALLSCRIPT="environment_macOS_inte.yml"
else
    echo "$OSTYPE with $ARCH not supported. Abort..."
    exit
fi
git remote set-url origin git@github.com:jscant/PointVS.git
conda env create -f $INSTALLSCRIPT
conda activate pointvs
pip install -e .
```
Note: Windows and Linux (arm64) are not supported, and Apple
Silicon MPS support will be enabled once
https://github.com/pytorch/pytorch/issues/77794 is resolved.

To run tests for invariance under E(3) transformations:
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
python3 point_vs.py multitask /tmp/test_run --model_task both \
-ea 1 -ep 1 --layers 3 \
--train_data_root_pose data/small_chembl_test \
--train_types_pose data/small_chembl_test.types \
--train_data_root_affinity data/multi_classification_sample \
--train_types_affinity data/multi_classification_sample.types \
--test_data_root_pose data/small_chembl_test \
--test_types_pose data/small_chembl_test.types \
--test_types_affinity data/multi_classification_sample.types \
--test_data_root_affinity data/multi_classification_sample
```

## Usage

```
usage: point_vs.py [-h] [--train_data_root_pose TRAIN_DATA_ROOT_POSE]
                   [--train_data_root_affinity TRAIN_DATA_ROOT_AFFINITY]
                   [--test_data_root_pose TEST_DATA_ROOT_POSE]
                   [--test_data_root_affinity TEST_DATA_ROOT_AFFINITY]
                   [--logging_level LOGGING_LEVEL] [--load_weights LOAD_WEIGHTS]
                   [--test_data_root TEST_DATA_ROOT]
                   [--translated_actives TRANSLATED_ACTIVES]
                   [--batch_size BATCH_SIZE] [--epochs_pose EPOCHS_POSE]
                   [--epochs_affinity EPOCHS_AFFINITY] [--channels CHANNELS]
                   [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                   [--wandb_project WANDB_PROJECT] [--wandb_run WANDB_RUN]
                   [--layers LAYERS] [--radius RADIUS] [--load_args LOAD_ARGS]
                   [--double] [--activation ACTIVATION] [--dropout DROPOUT]
                   [--use_1cycle] [--warm_restarts]
                   [--fourier_features FOURIER_FEATURES] [--norm_coords]
                   [--norm_feats] [--use_atomic_numbers] [--compact] [--thin_mlps]
                   [--hydrogens] [--augmented_actives AUGMENTED_ACTIVES]
                   [--min_aug_angle MIN_AUG_ANGLE]
                   [--max_active_rmsd MAX_ACTIVE_RMSD]
                   [--min_inactive_rmsd MIN_INACTIVE_RMSD] [--val_on_epoch_end]
                   [--synth_pharm] [--input_suffix INPUT_SUFFIX]
                   [--train_types_pose TRAIN_TYPES_POSE]
                   [--train_types_affinity TRAIN_TYPES_AFFINITY]
                   [--test_types_pose TEST_TYPES_POSE]
                   [--test_types_affinity TEST_TYPES_AFFINITY] [--egnn_attention]
                   [--egnn_tanh] [--egnn_normalise] [--egnn_residual]
                   [--edge_radius EDGE_RADIUS] [--end_flag] [--wandb_dir WANDB_DIR]
                   [--estimate_bonds] [--linear_gap] [--prune] [--top1] [--graphnorm]
                   [--multi_fc] [--lucid_node_final_act]
                   [--p_remove_entity P_REMOVE_ENTITY] [--static_coords]
                   [--permutation_invariance] [--node_attention]
                   [--attention_activation_function ATTENTION_ACTIVATION_FUNCTION]
                   [--node_attention_final_only] [--edge_attention_final_only]
                   [--node_attention_first_only] [--edge_attention_first_only]
                   [--only_save_best_models] [--egnn_edge_residual]
                   [--gated_residual] [--rezero] [--extended_atom_types]
                   [--max_inactive_rmsd MAX_INACTIVE_RMSD] [--model_task MODEL_TASK]
                   [--synthpharm] [--p_noise P_NOISE] [--include_strain_info]
                   [--final_softplus] [--optimiser OPTIMISER]
                   [--multi_target_affinity]
                   model save_path

positional arguments:
  model                 Type of point cloud network to use: lucid or egnn
  save_path             Directory in which experiment outputs are stored. If
                        wandb_run and wandb_project are specified,
                        save_path/wandb_project/wandb_run will be used to store
                        results.

optional arguments:
  -h, --help            show this help message and exit
  --train_data_root_pose TRAIN_DATA_ROOT_POSE
                        Location relative to which parquets files for training the
                        pose classifier as specified in the train_types_pose file are
                        stored.
  --train_data_root_affinity TRAIN_DATA_ROOT_AFFINITY, --tdra TRAIN_DATA_ROOT_AFFINITY
                        Location relative to which parquets files for training the
                        affinity predictor as specified in the train_types file are
                        stored.
  --test_data_root_pose TEST_DATA_ROOT_POSE
                        Location relative to which parquets files for testing the
                        pose classifier as specified in the test_types_pose file are
                        stored.
  --test_data_root_affinity TEST_DATA_ROOT_AFFINITY
                        Location relative to which parquets files for testing the
                        affinity predictor as specified in the test_types file are
                        stored.
  --logging_level LOGGING_LEVEL
                        Level at which to print logging statements. Any of notset,
                        debug, info, warning, error, critical.
  --load_weights LOAD_WEIGHTS, -l LOAD_WEIGHTS
                        Load a model.
  --test_data_root TEST_DATA_ROOT, -t TEST_DATA_ROOT
                        Location of structure test *.parquets files. Receptors should
                        be in a directory named receptors, with ligands located in
                        their specific receptor subdirectory under the ligands
                        directory.
  --translated_actives TRANSLATED_ACTIVES
                        Directory in which translated actives are stored. If
                        unspecified, no translated actives will be used. The use of
                        translated actives are is discussed in
                        https://pubs.acs.org/doi/10.1021/acs.jcim.0c00263
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Number of examples to include in each batch for training.
  --epochs_pose EPOCHS_POSE, -ep EPOCHS_POSE
                        Number of times to iterate through pose training set.
  --epochs_affinity EPOCHS_AFFINITY, -ea EPOCHS_AFFINITY
                        Number of times to iterate through affinity training set.
  --channels CHANNELS, -k CHANNELS
                        Channels for feature vectors
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate for gradient descent
  --weight_decay WEIGHT_DECAY, -w WEIGHT_DECAY
                        Weight decay for regularisation
  --wandb_project WANDB_PROJECT
                        Name of wandb project. If left blank, wandb logging will not
                        be used.
  --wandb_run WANDB_RUN
                        Name of run for wandb logging.
  --layers LAYERS       Number of layers in LieResNet
  --radius RADIUS       Maximum distance from a ligand atom for a receptor atom to be
                        included in input
  --load_args LOAD_ARGS
                        Load yaml file with command line args. Any args specified in
                        the file will overwrite other args specified on the command
                        line.
  --double              Use 64-bit floating point precision
  --activation ACTIVATION
                        Activation function
  --dropout DROPOUT     Chance for nodes to be inactivated on each trainin batch
                        (EGNN)
  --use_1cycle          Use 1cycle learning rate scheduling
  --warm_restarts       Use cosine annealing with warm restarts
  --fourier_features FOURIER_FEATURES
                        (Lucid) Number of fourier terms to use when encoding
                        distances (default is not to use fourier distance encoding)
  --norm_coords         (Lucid) Normalise coordinate vectors
  --norm_feats          (Lucid) Normalise feature vectors
  --use_atomic_numbers  Use atomic numbers rather than smina types
  --compact             Use compact rather than true one-hot encodings
  --thin_mlps           (Lucid) Use single layer MLPs for edge, node and coord
                        updates
  --hydrogens           Include polar hydrogens
  --augmented_actives AUGMENTED_ACTIVES
                        Number of randomly rotated actives to be included as decoys
                        during training
  --min_aug_angle MIN_AUG_ANGLE
                        Minimum angle of rotation for augmented actives as specified
                        in the augmented_actives argument
  --max_active_rmsd MAX_ACTIVE_RMSD
                        (Pose selection) maximum non-aligned RMSD between the
                        original crystal pose and active redocked poses
  --min_inactive_rmsd MIN_INACTIVE_RMSD
                        (Pose selection) minimum non-aligned RMSD between original
                        crystal pose and inactive redocked poses
  --val_on_epoch_end, -v
                        Run inference ion the validation set at the end of every
                        epoch during training
  --synth_pharm, -p     Synthetic Pharmacophore mode (for Tom, beta)
  --input_suffix INPUT_SUFFIX, -s INPUT_SUFFIX
                        Filename extension for inputs
  --train_types_pose TRAIN_TYPES_POSE
                        Optional name of GNINA-like types file which contains paths
                        and labels for a pose training set. See GNINA 1.0
                        documentation for specification.
  --train_types_affinity TRAIN_TYPES_AFFINITY
                        Optional name of GNINA-like types file which contains paths
                        and labels for an affinity training set. See GNINA 1.0
                        documentation for specification.
  --test_types_pose TEST_TYPES_POSE
                        Optional name of GNINA-like types file which contains paths
                        and labels for a pose test set. See GNINA 1.0 documentation
                        for specification.
  --test_types_affinity TEST_TYPES_AFFINITY
                        Optional name of GNINA-like types file which contains paths
                        and labels for an affinity test set. See GNINA 1.0
                        documentation for specification.
  --egnn_attention      Use attention mechanism on edges for EGNN
  --egnn_tanh           Put tanh layer at the end of the coordinates mlp (EGNN)
  --egnn_normalise      Normalise radial coordinates (EGNN)
  --egnn_residual       Use residual connections (EGNN)
  --edge_radius EDGE_RADIUS
                        Maximum interatomic distance for an edge to exist (EGNN)
  --end_flag            Add a file named "_FINISHED" to the save_path upon training
                        and test completion
  --wandb_dir WANDB_DIR
                        Location to store wandb files. Defaults to
                        <save_path>/<wandb_project>/<wandb_run>/wandb.
  --estimate_bonds      (EGNN): Instead of using a fixed edge radius,the
                        intermolecular radius is set at --edge_radius Angstroms but
                        the intramolecular radius is set at 2A, which has the effect
                        of putting edges where there are covalent bonds between atoms
                        in the same molecule.
  --linear_gap          Final linear layer comes after rather than before the global
                        average pooling layer. This can improve performance
                        significantly.
  --prune               (EGNN) Prune subgraphs which are not connected to the ligand
  --top1                A poorly kept secret ;)
  --graphnorm           (EGNN) add GraphNorm layers to each node MLP
  --multi_fc            Three fully connected layers rather than just one to
                        summarise the graph at the end of the EGNN
  --lucid_node_final_act
                        (Lucid) SiLU at the end of node MLPs
  --p_remove_entity P_REMOVE_ENTITY
                        Rate at which one of (randomly selected) ligand or receptor
                        is removed and label is forced to zero
  --static_coords       Do not update coords (eq. 4, EGNN)
  --permutation_invariance
                        Edge features are invariant to order of input node (EGNN,
                        experimental)
  --node_attention      Use attention mechanism for nodes
  --attention_activation_function ATTENTION_ACTIVATION_FUNCTION
                        One of sigmoid, relu, silu or tanh
  --node_attention_final_only
                        Only apply attention mechanism to nodes in the final layer
  --edge_attention_final_only
                        Only apply attention mechanism to edges in the final layer
  --node_attention_first_only
                        Only apply attention mechanism to nodes in the first layer
  --edge_attention_first_only
                        Only apply attention mechanism to edges in the first layer
  --only_save_best_models
                        Only save models which improve upon previous models
  --egnn_edge_residual  Residual connections for individual messages (EGNN)
  --gated_residual      Residual connections are gated by a single learnable
                        parameter (EGNN), see
                        home.ttic.edu/~savarese/savarese_files/Residual_Gates.pdf
  --rezero              ReZero residual connections (EGNN), see
                        arxiv.org/pdf/2003.04887.pdf
  --extended_atom_types
                        18 atom types rather than 10
  --max_inactive_rmsd MAX_INACTIVE_RMSD
                        Discard structures beyond <x> RMSD from xtal pose
  --model_task MODEL_TASK
                        One of either classification or regression;
  --synthpharm          For tom
  --p_noise P_NOISE     Probability of label being inverted during training
  --include_strain_info
                        Include info on strain energy and RMSD from ground state of
                        ligand
  --final_softplus      Final layer in regression has softplus nonlinearity
  --optimiser OPTIMISER, -o OPTIMISER
                        Optimiser (either adam or sgd)
  --multi_target_affinity
                        Use multitarget regression for affinity. If True, targets are
                        split depending on if labels are pkd, pki or IC50.

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
are used extensively in [GNINA 1.0](https://github.com/gnina/gnina).

Types File
Each line should follow one of two formats.

For classification datasets:
`<label> <energy> <rmsd> <receptor> <ligand>`

For regression datasets:
`<pki> <pkd> <ic50> <receptor> <ligand>`

where:

`<label>` is a binary label (0 or 1)

`<energy>` is usually set to -1.0, and can be used to store the energy of
an interaction or a docking score. It is included in the original GNINA
specification so is retained here.

`<rmsd>` is the RMSD from the original crystal structure where the ligand is a 
docked structure and the RMSD is known, or -1 otherwise.

`<pki>` The pKi, if known, else -1.0

`<pkd>` The pKd, if known, else -1.0

`<ic50>` The IC50, if known, else -1.0.

`<receptor>` is the relative location of the receptor structure in parquet format

`<ligand>` is the relative location of the ligand structure in parquet format

Only one of <pKi>, <pkd> or <ic50> should have a positive  value. If multiple types
of affinity are known, they can be given separate entries.

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
