#include <petsc.h>

int main(int argc, char **argv) { // I believe this code runs once per rank/process
  PetscMPIInt    rank, size;
  PetscInt       i;
  PetscInt       N;
  PetscReal      x, localval, localsum, globalsum;

  PetscCall(PetscInitialize(&argc,&argv,NULL,
      "Compute exp(x) in parallel with PETSc.\n\n"));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  // read option
  PetscOptionsBegin(PETSC_COMM_WORLD,"","options for expx","");
  PetscCall(PetscOptionsReal("-x","input to exp(x) function",NULL,x,&x,NULL));
  PetscCall(PetscOptionsInt("-N","number of terms to use",NULL,N,&N,NULL)); 
  // PetscOptionsReal(const char opt[], const char text[], const char man[], PetscReal currentvalue, PetscReal *value, PetscBool *set)

  PetscOptionsEnd();

  // compute  x^n/n!  where n = (rank of process) + 1
  //   localval = 1.0;
  //   for (i = 1; i < rank+1; i++)
  //       localval *= x/i;

  // Batch
  PetscInt batch_q = N / size;
  PetscInt batch_r = N % size; // remainder is the N - (largest multiple of batch_q and nP that is < N) and will always be < N

  PetscInt startterm = 1;
  PetscInt endterm = 1;

  if (rank < batch_r) {
    batch_q += 1;
    startterm = rank*batch_q;
    endterm = startterm+batch_q;
  }
  else {
    startterm = rank*batch_q+batch_r;
    endterm = startterm+batch_q;
  }

  localval = 1.0;
  localsum = 0.0;
  PetscReal absx = fabs(x);
  for (i = 1; i < endterm; i++) {
    localval *= absx/i;
    if (i >= startterm) {
        localsum += localval;
    }
  }

  if (rank == 0) {
    localsum += 1.0;
  }

  // sum the contributions over all processes
  PetscCall(MPI_Allreduce(&localsum,&globalsum,1,MPIU_REAL,MPIU_SUM,
      PETSC_COMM_WORLD));

  // output estimate and report on work from each process
  if (x<0){
    globalsum = 1 / globalsum;
  }
    
//   globalsum /= PETSC_MACHINE_EPSILON;

  PetscReal expx_true = exp(x);
  PetscReal rerror = fabs(globalsum - expx_true) / expx_true;
  rerror = rerror / PETSC_MACHINE_EPSILON;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "exp(%17.15f) is about %17.15f\n",x,globalsum));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "Relative error from C exp() is about %17.15f machine precisions\n",rerror));
//   PetscCall(PetscPrintf(PETSC_COMM_SELF,
//       "rank %d did %d flops\n",rank,(rank > 0) ? 2*rank : 0));
  PetscCall(PetscFinalize());
  return 0;
}
