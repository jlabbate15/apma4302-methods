#!/bin/bash

set -e  # Exit on any error
set -u  # Exit on undefined variable

# Script description
echo "Starting APMA 4302 PETSc configuration..."

export PETSC_DIR=$(pwd)
export PETSC_ARCH=apma4302-pkgs-opt

./configure \
  --with-cc=mpicc \
  --with-cxx=mpicxx \
  --with-fc=mpif90 \
  --with-debugging=0 \
  --with-shared-libraries=1 \
  --COPTFLAGS="-O3 -march=native" \
  --CXXOPTFLAGS="-O3 -march=native" \
  --FOPTFLAGS="-O3 -march=native" \
  --download-metis \
  --download-parmetis \
  --download-hypre \
  --download-scalapack \
  --download-mumps \
  --download-petsc4py \
  --with-hdf5-dir=$(brew --prefix hdf5-mpi) 


echo "Configuration completed successfully!"