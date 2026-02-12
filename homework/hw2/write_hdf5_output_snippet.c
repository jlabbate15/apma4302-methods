
#include <petscviewerhdf5.h>



    // output the solution, rhs, and exact solution to an HDF5 file    
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "bvp_solution.h5", FILE_MODE_WRITE, &viewer));
    PetscCall(PetscObjectSetName((PetscObject) uexact, "uexact"));
    PetscCall(PetscObjectSetName((PetscObject) f, "f"));
    PetscCall(PetscObjectSetName((PetscObject) u, "u"));
    PetscCall(VecView(f, viewer));
    PetscCall(VecView(u, viewer));
    PetscCall(VecView(uexact, viewer));
    PetscCall(PetscViewerDestroy(&viewer));


