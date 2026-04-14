# Notes for installing petsc

Most of these notes reflect instructions for macos (on arm64 architectures).  Your mileage may vary

## Preliminaries

Have installed:

* Compilers: gcc or Xcode
* git
* python 3 (a reasonably recent version, or use anaconda)
* a package manager
  * homebrew on macs
  * apt on linux machines
* Vscode
* XQuartz (X11 server) <https://www.xquartz.org/>
* Setup your private course repository on github (or gitlab) with the name `apma4302_<uni>` and add me as a collaborator.

## install key packages

* gfortran on Macs
  * `brew install gcc`
* openmpi
  * macos: `brew install open-mpi`
  * linux:
    * `sudo apt update`
    * `sudo apt install openmpi-bin openmpi-doc libopenmpi-dev`

## clone petsc repository from gitlab

* Instructions at <https://petsc.org/release/install/download/#recommended-obtain-release-version-with-git>)

## clone (or fork) my apma4302 repository from github

* `git clone https://github.com/mspieg/apma4302-methods.git`

## clone (or fork) the Beuler codes

* `git clone https://github.com/bueler/p4pdes.git`

## Follow along

* Build and check basic debuggable petsc
* Build and check basic optimized petsc
* Consider additional packages
* make streams
* Start with Ch1 of Beuler (build parallel e)
* Discuss homework

## Building with additional packages

* For more advances problems we will want some additional packages
* The installation script apma4302_configure-pkgs-opt.sh will also build
  * MUMPS (Multi frontal Massively Parallel solver) for parallel sparse direct
  * METIS/PARMETIS for parallel graph partitioning
  * HYPRE a robust algebraic multigrid package
  * petsc4py: python bindings for petsc and for interacting with petsc c codes with python
  * HDF5:  An advanced parallel and portable IO system for that can be imported in various visualization packages (and python)

## Building Firedrake

* To configure and build petsc for use with Firedrake, we will roughly follow the instructions at <https://www.firedrakeproject.org/firedrake/install.html>
  * **Add some packages with `brew`** (on MacOS)
    * basic instructions are included in the install script
   `arch-firedrake-default.sh`
    * temporarily disable brew autoupdate using
  `export HOMEBREW_NO_AUTO_UPDATE=1` in your shell
    * `brew install install bison fftw hwloc`
  * **clone your current anaconda environment**
    * either use the anaconda navigator or
    * `conda create --name firedrake --clone OLD_ENV_NAME`
    where `OLD_ENV_NAME` could be `base` or any environment your happy with but python version should be around 3.13
    * `conda activate firedrake`
  * **Build a new petsc arch**
    * move `arch-firedrake-default.sh` to your base petsc directory
    * build a new petsc with

      ```bash:
      cd $PETSC_DIR
      . ./arch-firedrake-default.sh
      make all
      make check

  * **set up your environment for PETSc**
    * You might need to add a path for `bison`
  * **Install firedrake into your firedrake installation**
    * download two firedrake repos into wherever you keep your repos

      ``` bash
      git clone https://github.com/firedrakeproject/firedrake.git .
      git clone -b OptionsManager_fix https://github.com/mspieg/firedrake-ts.git
    * pip install `firedrake` and `firedrake-ts` from the repos

      * starting in the directory that contain the two repos

      ```bash
      python -m pip install ./firedrake
      python -m pip install ./firedrake-ts

  * Cross your fingers that everything is working: go look at the example scripts e.g. `examples/firedrake/hemlholtz/helmholtz.py`
