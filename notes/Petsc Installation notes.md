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
    * `brew install bison fftw hwloc pnetcdf zlib ninja`
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
  
  
    
  * **pip install `firedrake` and `firedrake-ts` from the repos**

    * starting in the directory that contain the two repos

    ```bash
    python -m pip install ./firedrake
    python -m pip install ./firedrake-ts

* Cross your fingers that everything is working: go look at the example scripts e.g. `examples/firedrake/hemlholtz/helmholtz.py`

## new instructions for installing Firedrake

Since I first built Firedrake and Firedrake-ts, there have been modifications to both PETSc and Firedrake and it makes some sense to rebuild against the latest stable releases of both.  So do the following

* **update brew packages** (Macos)
  Use instructions above

* **create empty firedrake conda environment** with python 3.13 (if using anaconda)

  ``` bash
  conda create -n firedrake-new python=3.13.13
  conda activate firedrake-new
  pip install numpy vtk

* **update petsc to latest release** and rebuild  petsc with the new script **`arch-firedrake-default.sh`** 

    ```bash
    cp <path-to>apma4302-methods/installation_scripts/arch-firedrake-default.sh ${PETSC_DIR}

* You will first need to make sure an up-to-date bison is in your path.  If you used brew do

    ```bash
    export PATH=/opt/homebrew/opt/bison/bin:$PATH
    cd $PETSC_DIR
    git checkout release
    git pull
    . ./arch-firedrake-default.sh
    make all
    make check

* This new version will also build a compatible `petsc4py` as well as a  shell script `set_firedrake_pestc_env.sh` to set your environment variables and `PYTHONPATH`

  ```bash
    . ./set_firedrake_petsc_env.sh

* **install firedrake 2026.4 from the web**

  ```bash
  pip cache purge
  pip install firedrake
* **update your firedrake-ts to the latest fix and install**

    ```bash
    cd <path-to>/firedrake-ts
    git fetch origin
    git checkout 2026.4-fixes
    git pull
    cd ..
    python -m pip install ./firedrake-ts

* **And test against firedrake examples in this repo**

  ```bash
  cd <path-to>/apma4302-methods/examples/firedrake
  cd helmholtz
  firedrake-clean
  python helmholtz-snes
  cd ../heatflow
  python heat.py



