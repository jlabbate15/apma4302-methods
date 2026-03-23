static char help[] = "2D reaction-diffusion problem with DMDA and SNES.  Option prefix -rct_.\n\n";

// run with mpirun -n 4 ./reaction2d -options_file options_file

#include <petsc.h>

typedef struct {
    PetscReal  gamma;
    PetscInt  p;
    PetscBool  rct_linear_f;
} AppCtx;

// extern PetscReal f_source(PetscReal);
// extern PetscErrorCode InitialAndExact(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
// extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
// extern PetscErrorCode formRHS(DM, Vec);
// extern PetscErrorCode formMatrix(DM, Mat);
// extern only needed if functions are defined in another file
PetscErrorCode formExact(DM, Vec);
PetscErrorCode formf(DM, Vec, AppCtx *);
static PetscReal f_MMS(PetscReal x, PetscReal y, AppCtx *user);
PetscReal ufunction(PetscReal x, PetscReal y);
PetscReal d2ufunction(PetscReal x, PetscReal y);
PetscErrorCode formFunctionLocal(DMDALocalInfo *, PetscReal **, PetscReal **, void *);
PetscErrorCode FormJacobianLocal(DMDALocalInfo *, PetscReal **, Mat, Mat, void *);

//STARTMAIN
int main(int argc,char **args) {
  DM            da;
  SNES          snes;
  AppCtx        user;
  Vec           u, uexact, f, ustore, u_rel;
  PetscReal     errnorm, uexactnorm, errnorm2;
  PetscInt     M,N;
  DMDALocalInfo info;
  PetscViewer   viewer;
  PetscMPIInt   rank;


  // default values
  user.rct_linear_f = PETSC_FALSE;
  user.gamma = 1.0;
  user.p = 2.0;
  M = 10; // default to 10x10 grid
  N = 10; // default to 10x10 grid

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));


  // set inputs
  PetscOptionsBegin(PETSC_COMM_WORLD,"rct_","options for reaction2d",""); 
  PetscCall(PetscOptionsBool("-linear_f","set right hand side","reaction2d.c",user.rct_linear_f,&(user.rct_linear_f),NULL));
  PetscCall(PetscOptionsReal("-gamma","value of gamma","reaction2d.c",user.gamma,&(user.gamma),NULL));
  PetscCall(PetscOptionsInt("-p","value of p","reaction2d.c",user.p,&(user.p),NULL));
  PetscOptionsEnd();

  // create DMDA object for 2D, not 1D
  //   PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,9,1,1,NULL,&da));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da)); // DoF is how many values you have per node point, so for a scalar field this = 1
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0,1, 0,1, 0,1));
  PetscCall(DMSetApplicationContext(da,&user));

  PetscCall(DMCreateGlobalVector(da,&u));
  PetscCall(VecDuplicate(u,&uexact));
  PetscCall(VecDuplicate(u,&f));
  PetscCall(VecDuplicate(u,&ustore));
  PetscCall(VecDuplicate(u,&u_rel));


  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(formExact(da,uexact));
  PetscCall(VecCopy(uexact, u)); // copy uexact into u
  PetscCall(formf(da,f,&user));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,da));
  PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunctionFn *)formFunctionLocal,&user));
  PetscCall(DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobianFn *)FormJacobianLocal,&user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes,NULL,u));

  PetscCall(VecCopy(u, ustore));
  PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uexact
  PetscCall(VecNorm(u,NORM_INFINITY,&errnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "on %d x %d point grid:  |u-u_exact|_inf = %g\n",info.mx,info.my,errnorm));

  PetscCall(VecCopy(ustore, u_rel));
  PetscCall(VecNorm(uexact,NORM_2,&uexactnorm));
  PetscCall(VecAXPY(u_rel,-1.0,uexact));    // u_rel <- u_rel + (-1.0) uexact
  PetscCall(VecNorm(u_rel,NORM_2,&errnorm2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "on %d x %d point grid:  |u-u_exact|_2 / |u_exact|_2 = %g\n",info.mx,info.my,errnorm2/uexactnorm));


  //output vectors to a VTK file for visualization in Paraview
  PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD,"reaction2d.vtr",FILE_MODE_WRITE,&viewer));
  PetscCall(PetscObjectSetName((PetscObject) uexact, "uexact"));
  PetscCall(PetscObjectSetName((PetscObject) ustore, "u"));
  PetscCall(PetscObjectSetName((PetscObject) f, "f"));
//   PetscCall(PetscObjectSetName((PetscObject) b, "b"));  
//   PetscCall(PetscObjectSetName((PetscObject) rankmap, "rankmap"));
  PetscCall(VecView(uexact, viewer));
  PetscCall(VecView(ustore, viewer));
  PetscCall(VecView(f, viewer));
//   PetscCall(VecView(b, viewer));
//   PetscCall(VecView(rankmap, viewer));
  PetscCall(DMView(da, viewer));
  PetscCall(PetscViewerDestroy(&viewer));


  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&u_rel));
  PetscCall(VecDestroy(&ustore));
  PetscCall(VecDestroy(&uexact));
  PetscCall(VecDestroy(&f));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
//ENDMAIN


PetscReal ufunction(PetscReal x, PetscReal y) { // uexact
    PetscReal sigma = 0.3;
    PetscReal x0 = 0.65, y0 = 0.65;
    PetscReal r2 = (x-x0)*(x-x0) + (y-y0)*(y-y0);
    PetscReal amp = 1.0;
    return amp * PetscExpReal( - r2  / (sigma*sigma) );
    //return x*x * (1.0 - x*x) * y*y * (1.0 - y*y);
}

PetscReal d2ufunction(PetscReal x, PetscReal y) { // Laplacian of uexact evaluated at one point (x,y)
    PetscReal sigma = 0.3;
    PetscReal x0 = 0.65, y0 = 0.65;
    PetscReal amp = 1.0;
    PetscReal r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
    PetscReal expterm = PetscExpReal(-r2 / (sigma * sigma));
    return amp * expterm * 4.0 / (sigma * sigma) * (r2 / (sigma * sigma) - 1.0);
}

/* MMS forcing for -lap(u) + gamma*u^p = f (same as formf uses) */
static PetscReal f_MMS(PetscReal x, PetscReal y, AppCtx *user)
{ // returns f, the right hand side of the equation, at one point (x,y)
    PetscReal lap = d2ufunction(x, y);
    if (user->rct_linear_f) return -lap;
    return -lap + user->gamma * PetscPowReal(ufunction(x, y), (PetscReal)user->p);
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscReal **u, Mat J, Mat P, void *ctx)
{ // forms Jacobian of the residual F(u)
    AppCtx       *user = (AppCtx *)ctx;
    MatStencil    row, col[5];
    PetscReal     hx, hy, v[5];
    PetscInt      i, j, ncols;

    hx = 1.0 / (info->mx - 1);
    hy = 1.0 / (info->my - 1);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            row.j = j;
            row.i = i;
            col[0].j = j;
            col[0].i = i;
            ncols = 1;
            if (i == 0 || i == info->mx - 1 || j == 0 || j == info->my - 1) { // boundary points
                v[0] = 1.0;
            } else {
                /* Same 5-point -Laplacian as poisson2d formMatrix + hx*hy * d(γ u^p)/du */
                v[0] = 2.0 * (hy / hx + hx / hy) + hx * hy * user->gamma * (PetscReal)user->p * PetscPowReal(u[j][i], (PetscReal)user->p - 1.0);
                if (i - 1 > 0) {
                    col[ncols].j = j;
                    col[ncols].i = i - 1;
                    v[ncols++] = -hy / hx;
                }
                if (i + 1 < info->mx - 1) {
                    col[ncols].j = j;
                    col[ncols].i = i + 1;
                    v[ncols++] = -hy / hx;
                }
                if (j - 1 > 0) {
                    col[ncols].j = j - 1;
                    col[ncols].i = i;
                    v[ncols++] = -hx / hy;
                }
                if (j + 1 < info->my - 1) {
                    col[ncols].j = j + 1;
                    col[ncols].i = i;
                    v[ncols++] = -hx / hy;
                }
            }
            PetscCall(MatSetValuesStencil(P, 1, &row, ncols, col, v, INSERT_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
    if (J != P) PetscCall(MatCopy(P, J, SAME_NONZERO_PATTERN));
    return 0;
}

//STARTEXACT
PetscErrorCode formExact(DM da, Vec uexact) { // forms exact solution uexact using helper functions above
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **auexact;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    PetscCall(DMDAVecGetArray(da, uexact, &auexact)); // get a "2D" indexable array from the vector, like unflattening it
    for (j = info.ys; j < info.ys+info.ym; j++) {
        y = j * hy;
        for (i = info.xs; i < info.xs+info.xm; i++) {
            x = i * hx;
            auexact[j][i] = ufunction(x,y);
        }
    }
    PetscCall(DMDAVecRestoreArray(da, uexact, &auexact));
    return 0;
}
//ENDEXACT

//STARTRHS
// PetscErrorCode formRHS(DM da, Vec b) {
//     PetscInt       i, j;
//     PetscReal      hx, hy, x, y, **ab;
//     DMDALocalInfo  info;

//     PetscCall(DMDAGetLocalInfo(da,&info));
//     hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
//     PetscCall(DMDAVecGetArray(da, b, &ab));
//     for (j = info.ys; j < info.ys + info.ym; j++)
//     {
//         y = j * hy;
//         for (i = info.xs; i < info.xs + info.xm; i++)
//         {
//             x = i * hx;
//             if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1)
//             {
//                 ab[j][i] = ufunction(x, y); // on boundary: 1*u = uexact(x,y)
//             }
//             else
//             {
//                 ab[j][i] = -hx * hy * d2ufunction(x, y); // interior: f(x_i,y_j)
//                 // // lift dirichlet BCs into interior equations
//                 if (i == 1)
//                 {
//                     ab[j][i] += hy / hx * ufunction(x - hx, y);
//                 }
//                 if (i == info.mx - 2)
//                 {
//                     ab[j][i] += hy / hx * ufunction(x + hx, y);
//                 }
//                 if (j == 1)
//                 {
//                     ab[j][i] += hx / hy * ufunction(x, y - hy);
//                 }
//                 if (j == info.my - 2)
//                 {
//                     ab[j][i] += hx / hy * ufunction(x, y + hy);
//                 }
//             }
//         }
//     }
//     PetscCall(DMDAVecRestoreArray(da, b, &ab));
//     return 0;
// }
//ENDRHS


//STARTMATRIX
// PetscErrorCode formMatrix(DM da, Mat A) {
//     DMDALocalInfo  info;
//     MatStencil     row, col[5];
//     PetscReal      hx, hy, v[5];
//     PetscInt       i, j, ncols;

//     PetscCall(DMDAGetLocalInfo(da,&info));
//     hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
//     for (j = info.ys; j < info.ys+info.ym; j++) {
//         for (i = info.xs; i < info.xs+info.xm; i++) {
//             row.j = j;           // row of A corresponding to (x_i,y_j)
//             row.i = i;
//             col[0].j = j;        // diagonal entry
//             col[0].i = i;
//             ncols = 1;
//             if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
//                 v[0] = 1.0;      // on boundary: trivial equation
//             } else {
//                 v[0] = 2.*(hy/hx + hx/hy); // interior: build a row
//                 if (i-1 > 0) {
//                     col[ncols].j = j;    col[ncols].i = i-1;
//                     v[ncols++] = -hy/hx;
//                 }
//                 if (i+1 < info.mx-1) {
//                     col[ncols].j = j;    col[ncols].i = i+1;
//                     v[ncols++] = -hy/hx;
//                 }
//                 if (j-1 > 0) {
//                     col[ncols].j = j-1;  col[ncols].i = i;
//                     v[ncols++] = -hx/hy;
//                 }
//                 if (j+1 < info.my-1) {
//                     col[ncols].j = j+1;  col[ncols].i = i;
//                     v[ncols++] = -hx/hy;
//                 }
//             }
//             PetscCall(MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES));
//         }
//     }
//     PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
//     PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
//     return 0;
// }
//ENDMATRIX


//STARTf
PetscErrorCode formf(DM da, Vec f, AppCtx *user) { // forms f, the right hand side of the equation, for VTK visualization
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **af;
    DMDALocalInfo  info;
    PetscReal      gamma = user->gamma;
    PetscReal      p = user->p;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    PetscCall(DMDAVecGetArray(da, f, &af));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = i * hx;
            if (user->rct_linear_f) {
                af[j][i] = -d2ufunction(x, y);
            } else {
                af[j][i] = -d2ufunction(x, y) + gamma * PetscPowReal(ufunction(x, y), (PetscReal)p);
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(da, f, &af));
    return 0;
}
//ENDRHS

/*
 * Strong form: -Δu + γ u^p - f = 0,  f from MMS (continuous ∇²u in f_MMS).
 * Discrete interior: same 5-point Laplacian as poisson2d.c formMatrix, Dirichlet neighbors
 * substituted with u_exact (elimination). Reaction/source scaled by hx*hy so that at u_exact,
 * F ~ 0 up to truncation error (HW Eq.(4) hx-only scaling does not match continuous f).
 */
PetscErrorCode formFunctionLocal(DMDALocalInfo *info, PetscReal **u, PetscReal **FF, void *ctx)
{
    AppCtx   *user = (AppCtx *)ctx;
    PetscInt  i, j;
    PetscReal hx, hy, x, y, lap, fval, uw, ue, us, un;

    hx = 1.0 / (info->mx - 1);
    hy = 1.0 / (info->my - 1);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            if (i == 0 || i == info->mx - 1 || j == 0 || j == info->my - 1) {
                FF[j][i] = u[j][i] - ufunction(x, y);
            } else {
                uw = (i - 1 == 0) ? ufunction(x - hx, y) : u[j][i - 1];
                ue = (i + 1 == info->mx - 1) ? ufunction(x + hx, y) : u[j][i + 1];
                us = (j - 1 == 0) ? ufunction(x, y - hy) : u[j - 1][i];
                un = (j + 1 == info->my - 1) ? ufunction(x, y + hy) : u[j + 1][i];

                lap = 2.0 * (hy / hx + hx / hy) * u[j][i];
                lap -= (hy / hx) * (uw + ue);
                lap -= (hx / hy) * (us + un);
                fval = f_MMS(x, y, user);
                FF[j][i] = lap + hx * hy * user->gamma * PetscPowReal(u[j][i], (PetscReal)user->p) - hx * hy * fval;
            }
        }
    }
    return 0;
}


//ENDFUNCTIONS
