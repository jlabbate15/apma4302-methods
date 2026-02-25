//STARTWHOLE
static char help[] = "Solve a tridiagonal system of arbitrary size.\n, Option prefix = tri_.\n";

#include <petsc.h>
#include <petscviewerhdf5.h>


int main(int argc,char **args) {
    Vec         u, uexact, f, ures;
    Mat         A;
    KSP         ksp;
    PetscInt    m = 4, i, Istart, Iend;
    PetscReal   errnorm, unorm, rerr;
    PetscScalar h, xi, diag;
    PetscScalar gamma = 1.0, c = 1.0, k = 1.0;
    PetscViewer viewer;

    PetscCall(PetscInitialize(&argc,&args,NULL,help));

    PetscOptionsBegin(PETSC_COMM_WORLD,"bvp_","options for bvp",NULL); // specify prefix on variables in "options_file" to look for, here it is "bvp_"
    PetscCall(PetscOptionsInt("-m","dimension of linear system","tri.c",m,&m,NULL));
    // options_file is read automatically by PETSc
    // Additionally, the "bvp_" subscript is handled automatically by PETSc
    PetscCall(PetscOptionsReal("-gamma","gamma coefficient","bvp.c", gamma, &gamma, NULL));
    PetscCall(PetscOptionsReal("-c","constant in u","bvp.c", c, &c, NULL));
    PetscCall(PetscOptionsReal("-k","real value in u","bvp.c", k, &k, NULL));
    PetscOptionsEnd();

    h = 1.0 / m;

    PetscCall(VecCreate(PETSC_COMM_WORLD,&u)); // create vector x
    PetscCall(VecSetSizes(u,PETSC_DECIDE,m+1)); // set length of x to be m, PETSC_DECIDE auto-configures process splitting
    PetscCall(VecSetFromOptions(u)); // vector type set based on command line (CUDA, other), not totally sure what is happening here, but I believe it would come from the "options_file" input
    PetscCall(VecDuplicate(u,&f)); // copy x structure to f, not values
    // PetscCall(VecDuplicate(x,&b)); // copy x structure to b, not values
    PetscCall(VecDuplicate(u,&uexact)); // copy x structure to xexact, not values

    PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
    PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m+1,m+1));
    // PetscCall(MatSetOptionsPrefix(A,"a_"));
    PetscCall(MatSetFromOptions(A)); // matrix type set based on command line (CUDA, other), not totally sure what is happening here, but I believe it would come from the "options_file" input for anything with a "a_" prefix
    PetscCall(MatSetUp(A)); // final setup, must be called before setting values
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend)); // tells each process which rows it owns (assigns i indices to each processor for A(i,j))

    for (i=Istart; i<Iend; i++) { // different loop per processor
        xi = i * h;

        PetscScalar xval = PetscSinReal(k * PETSC_PI * xi) + c*PetscPowReal(xi-0.5,3);
        // }
        PetscCall(VecSetValues(uexact,1,&i,&xval,INSERT_VALUES)); // this is assembling the exact (or manufactured) solution, given by the problem statement

        PetscScalar fi = (k*k * PETSC_PI*PETSC_PI + gamma) * PetscSinReal(k * PETSC_PI * xi) + c * (gamma * PetscPowReal(xi - 0.5, 3) - 6.0 * (xi - 0.5));
        PetscCall(VecSetValues(f,1,&i,&fi,INSERT_VALUES));


        if (i == 0 || i == m) { // set diagonals, which are the bcs
            diag = 1.0;
            PetscCall(MatSetValues(A,1,&i,1,&i,&diag,INSERT_VALUES));
        } else { // set non-diagnols with stencil for -(u'')+gamma*u
            PetscInt idx[3] = {i-1,i,i+1}; // not doing i=0 or i=m so this is fine
            PetscScalar input[3] = {-1.0/(h*h),2.0/(h*h) + gamma,-1.0/(h*h)};
            PetscCall(MatSetValues(A,1,&i,3,idx,input,INSERT_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(uexact));
    PetscCall(VecAssemblyEnd(uexact));
    PetscCall(VecAssemblyBegin(f));
    PetscCall(VecAssemblyEnd(f));
    // PetscCall(MatMult(A,xexact,f));

    PetscInt bcRows[2] = {0,m};

    PetscCall(MatZeroRowsColumns(
        A,
        2,          // number of rows
        bcRows,     // which rows
        1.0,        // diagonal value to set
        uexact,     // vector containing correct bcs
        f           // RHS vector (gets modified)
    ));

    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetOperators(ksp,A,A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp,f,u));

    PetscCall(VecNorm(uexact,NORM_2,&unorm));
    PetscCall(VecDuplicate(u,&ures));
    PetscCall(VecCopy(u,ures));
    PetscCall(VecAXPY(ures,-1.0,uexact)); // x = x - xexact, notice x is set
    PetscCall(VecNorm(ures,NORM_2,&errnorm));
    rerr = errnorm / unorm;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "relative error for m = %d system is %.1e\n",m,rerr));

     // problem statement additions
    // output the solution, rhs, and exact solution to an HDF5 file
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "bvp_solution.h5",FILE_MODE_WRITE, &viewer));
    PetscCall(PetscObjectSetName((PetscObject) uexact, "uexact"));
    PetscCall(PetscObjectSetName((PetscObject) f, "f"));
    PetscCall(PetscObjectSetName((PetscObject) u, "u"));
    PetscCall(VecView(f, viewer));
    PetscCall(VecView(u, viewer));
    PetscCall(VecView(uexact, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&f));
    // PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&uexact));
    PetscCall(PetscFinalize());

   

    return 0;
}
//ENDWHOLE
