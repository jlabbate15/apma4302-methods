#!/bin/bash

# Script description
#
#   PETSc configuration script for the default Firedrake configuration that works on my M2 MacBook Pro.
#   where I have already installed apma4302-pkgs-opt
#   This is intended to be run from the root of the PETSc source directory, and will create a new PETSc configuration in a subdirectory called `arch-firedrake-default`.
#   First check to see if the following dependences are already installed in homebrew
#     - gcc
#     - open-mpi
#     - hdf5-mpi
#   if not install them along with the following dependencies:
#     - bison
#     - fftw
#     - hwloc
#   If you have not installed these dependencies, you can do so with the following command:
#     export HOMEBREW_NO_AUTO_UPDATE=1
#     brew install bison fftw hwloc 
#   Additional packages will be downloaded and built automatically by the PETSc configuration script.
#   Note that this script is intended to be a starting point for a PETSc configuration that works well with Firedrake. Depending on your specific needs and system configuration, 
#   you may need to modify the options passed to the `configure` script.
#
#   To run this script, simply execute it from the root of the PETSc source directory:
#
echo "Starting Firedrake PETSc configuration..."

export PETSC_DIR=$(pwd)
export PETSC_ARCH=arch-firedrake-default

./configure \
  --with-cc=mpicc \
  --with-cxx=mpicxx \
  --with-fc=mpif90 \
  --with-c2html=0 \
  --with-fortran-bindings=0 \
  --with-strict-petscerrorcode \
  --with-debugging=0 \
  --with-shared-libraries=1 \
  --COPTFLAGS="-O3 -march=native -mtune=native" \
  --CXXOPTFLAGS="-O3 -march=native -mtune=native" \
  --FOPTFLAGS="-O3 -march=native" \
  --with-bison-dir=/opt/homebrew/opt/bison \
  --with-fftw-dir=/opt/homebrew \
  --with-hwloc-dir=/opt/homebrew \
  --with-zlib-dir=/opt/homebrew/opt/zlib \
  --with-hdf5-dir=/opt/homebrew \
  --download-netcdf \
  --with-pnetcdf-dir=/opt/homebrew \
  --download-ptscotch \
  --download-suite-sparse \
  --download-superlu_dist \
  --download-metis \
  --download-parmetis \
  --download-hypre \
  --download-scalapack \
  --download-mumps-avoid-mpi-in-place \
  --download-mumps 

echo "Configuration completed successfully!"

#--with-c2html=0 --with-debugging=0 --with-fortran-bindings=0 --with-shared-libraries=1 --with-strict-petscerrorcode PETSC_ARCH=arch-firedrake-default --COPTFLAGS='-O3 -march=native -mtune=native' --CXXOPTFLAGS='-O3 -march=native -mtune=native' --FOPTFLAGS='-O3' -download-mumps-avoid-mpi-in-place --download-bison --with-fftw-dir=/opt/homebrew --with-hdf5-dir=/opt/homebrew --with-hwloc-dir=/opt/homebrew --with-metis-dir=/opt/homebrew --download-mumps --download-netcdf --with-pnetcdf-dir=/opt/homebrew --download-ptscotch --with-scalapack-dir=/opt/homebrew --with-suitesparse-dir=/opt/homebrew --download-superlu_dist --with-zlib-dir=/opt/homebrew/opt/zlib --download-hypre