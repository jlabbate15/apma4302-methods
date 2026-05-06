#!/bin/bash
#SBATCH --account=apam
#SBATCH --exclusive
#SBATCH --ntasks=64
#SBATCH --tasks-per-node=8
#SBATCH --time=120
#SBATCH --mail-user=USERNAME@columbia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=strong_fish_N257_P064_p08.o.%j

# set some environment variables for apptainer
module load singularity 
export APPTAINER_TMPDIR=$SINGULARITY_TMPDIR
export APPTAINER_BINDPATH=$SINGULARITY_BINDPATH
module load openmpi/gcc/64/4.1.7a1

# set the container for firedrake
SIF=/burg/home/mws6/sifs/firedrake-ts.sif

# Launches the MPI application.
GO="mpiexec -n $SLURM_NTASKS apptainer exec  $SIF"
cd $SLURM_SUBMIT_DIR

# FISH:  solve 3D Poisson equation
# using optimal CG+GMG solver
# -da_refine 7 is 257x257x257
# ~ 12 Gb memory
# coarse grid is 9x9x9 so should work up to several hundred processors
$GO ./fish -fsh_dim 3 -da_refine 7 -pc_mg_levels 6 -pc_type mg -ksp_type cg -snes_type ksponly -ksp_converged_reason -ksp_monitor -log_view

