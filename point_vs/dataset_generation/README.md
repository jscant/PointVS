#Dataset generation

Some of the more important scripts found in `pointvs/dataset_generation` are
explained here.

##generate_types_file.py

Generate a types file including RMSD calculations if specified from PDB, SDF
and MOL2 files.
```
usage: generate_types_file.py [-h] [--receptor_pattern RECEPTOR_PATTERN]
                              [--crystal_pose_pattern CRYSTAL_POSE_PATTERN]
                              [--docked_pose_pattern DOCKED_POSE_PATTERN]
                              [--active_pattern ACTIVE_PATTERN]
                              [--inactive_pattern INACTIVE_PATTERN]
                              base_path output_path

positional arguments:
  base_path             Root directory in which pdb, sdf and mol2 files to be
                        converted to parquets are stored
  output_path           Directory in which to store output parquets

optional arguments:
  -h, --help            show this help message and exit
  --receptor_pattern RECEPTOR_PATTERN, -r RECEPTOR_PATTERN
                        Regex pattern for receptor structure
  --crystal_pose_pattern CRYSTAL_POSE_PATTERN, -x CRYSTAL_POSE_PATTERN
                        Regex pattern for crystal poses
  --docked_pose_pattern DOCKED_POSE_PATTERN, -d DOCKED_POSE_PATTERN
                        Regex pattern for docked poses
  --active_pattern ACTIVE_PATTERN, -a ACTIVE_PATTERN
                        Regex pattern for active poses
  --inactive_pattern INACTIVE_PATTERN, -i INACTIVE_PATTERN
                        Regex pattern for inactive structures
```
The directory structure must be set up as follows:
In the base_path, there should be multiple folders each containing 1 or 0
PDB receptor files, alongside sdf/mol2 ligand files. There are two modes,
which are selected by specifying EITHER:

    ====== --crystal_pose_pattern AND --docked_pose_pattern ======
    Each pose found in sdf files using the --docked_pose_pattern expression will
    be compared to the structure found in the sdf matching the
    --crystal_pose_pattern, so the RMSD will be found and labels will be
    assigned based on whether the RMSD is above or below 2A.

    If there are multiple docked and crystal poses in each subdirectory, an
    attempt will be made to match each docked sdf with a crystal sdf, based on
    the largest common substring. If this cannot be done in a 1-to-1 manner,
    an exception will be thrown.

                             ==== OR ====

    ====== --active_pattern AND --inactive_pattern ======
    No RMSD will be calculated; structures matching the active and inactive
    patterns will be assigned labels 1 and 0 respectively.


##mol_to_parquet.py

```
usage: mol_to_parquet.py [-h] types_file output_path types_base_path

positional arguments:
  types_file       Input file, any of sdf, pdb or mol2 format accepted
  output_path      Directory in which to store resultant parquet files
  types_base_path  Root relative to which types file entries are made. This
                   should contain all of the SDF files to be converted.
```

This script will take the output of `generate_types_file.py` and convert the
pdb/sdf files used to generate the types file into parquets. The output
directory structure will that found in the types file, relative to
`output_path`.