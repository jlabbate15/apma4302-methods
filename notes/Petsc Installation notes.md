## Notes for installing petsc:

Most of these notes reflect instructions for macos (on arm64 architectures).  Your mileage may vary

### Preliminaries

Have installed:
* Compilers: gcc or Xcode
* git
* python 3
* a package manager 
    * homebrew on macs
    * apt on linux machines
* Vscode
* Setup your private course repository on github (or gitlab) with the name `apma4302_<uni>` and add me as a collaborator.

### install key packages
* gfortran on Macs 
    * `brew install gcc`
* openmpi
    * macos: `brew install open-mpi`
    * linux: 
        * `sudo apt update`
        *  `sudo apt install openmpi-bin openmpi-doc libopenmpi-dev`


### clone petsc repository from gitlab
* Instructions at https://petsc.org/release/install/download/#recommended-obtain-release-version-with-git)

### clone (or fork) my apma4302 repository from gitlab

* `git clone https://github.com/mspieg/apma4302-methods.git`

### clone (or fork) the Beuler codes

* `git clone https://github.com/bueler/p4pdes.git`

### Follow along

* Build and check basic debuggable petsc
* Build and check basic optimized petsc
* Consider additional packages
* make streams
* Start with Ch1 of Beuler (build parallel e)
* Discuss homework

