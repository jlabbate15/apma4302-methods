static char help[] = "A structured-grid Poisson solver using DMDA+KSP.\n\n";

#include <petsc.h>

extern PetscErrorCode formMatrix(DM, Mat);
extern PetscErrorCode formExact(DM, Vec);
extern PetscErrorCode formRHS(DM, Vec);
extern PetscErrorCode formF(DM, Vec);
extern PetscErrorCode formRankMap(DM, Vec, PetscInt);
extern PetscReal  ufunction(PetscReal, PetscReal);
extern PetscReal  d2ufunction(PetscReal, PetscReal);

//STARTMAIN
int main(int argc,char **args) {
    DM            da;
    Mat           A;
    Vec           b,u,uexact, rankmap, f;
    KSP           ksp;
    PetscReal     errnorm, uexactnorm;
    DMDALocalInfo info;
    PetscMPIInt   rank;
    PetscViewer     viewer;

    PetscCall(PetscInitialize(&argc,&args,NULL,help));

    //get rank
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    // change default 9x9 size using -da_grid_x M -da_grid_y N
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                 DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                 9,9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));

    // create linear system matrix A
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0));
    PetscCall(DMCreateMatrix(da,&A));
    PetscCall(MatSetFromOptions(A));

    // create RHS b, approx solution u, exact solution uexact
    PetscCall(DMCreateGlobalVector(da,&b));
    PetscCall(VecDuplicate(b,&u));
    PetscCall(VecDuplicate(b,&uexact));
    PetscCall(VecDuplicate(b,&rankmap));
    PetscCall(VecDuplicate(b,&f));


    // fill vectors and assemble linear system
    PetscCall(formExact(da,uexact));
    PetscCall(formF(da,f));
    PetscCall(formRHS(da,b));
    PetscCall(formMatrix(da,A));
    PetscCall(formRankMap(da,rankmap,(PetscInt) rank));

    // create and solve the linear system
    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetOperators(ksp,A,A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp,b,u));

    //output vectors to a VTK file for visualization in Paraview
    PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD,"poisson.vtr",FILE_MODE_WRITE,&viewer));
    PetscCall(PetscObjectSetName((PetscObject) uexact, "uexact"));
    PetscCall(PetscObjectSetName((PetscObject) u, "u"));
    PetscCall(PetscObjectSetName((PetscObject) f, "f"));
    PetscCall(PetscObjectSetName((PetscObject) b, "b"));  
    PetscCall(PetscObjectSetName((PetscObject) rankmap, "rankmap"));
    PetscCall(VecView(uexact, viewer));
    PetscCall(VecView(u, viewer));
    PetscCall(VecView(f, viewer));
    PetscCall(VecView(b, viewer));
    PetscCall(VecView(rankmap, viewer));
    PetscCall(DMView(da, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    // report on grid and numerical error
    PetscCall(VecNorm(uexact,NORM_2,&uexactnorm));
    PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uxact
    PetscCall(VecNorm(u,NORM_2,&errnorm));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "on %d x %d grid:  rel_error |u-uexact|_2/|uexact|_2 = %g\n",
                info.mx,info.my,errnorm/uexactnorm));
    
    // clean up and finalize 
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&uexact));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&f));
    PetscCall(VecDestroy(&rankmap));
    PetscCall(MatDestroy(&A));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(DMDestroy(&da));
    PetscCall(PetscFinalize());
    return 0;
}
//ENDMAIN

// exact solution and its Laplacian for off-center Gaussian bump
PetscReal ufunction(PetscReal x, PetscReal y) {
    PetscReal sigma = 0.3;
    PetscReal x0 = 0.65, y0 = 0.65;
    PetscReal r2 = (x-x0)*(x-x0) + (y-y0)*(y-y0);
    PetscReal amp = 1.0;
    return amp * PetscExpReal( - r2  / (sigma*sigma) );
    //return x*x * (1.0 - x*x) * y*y * (1.0 - y*y);
}

PetscReal d2ufunction(PetscReal x, PetscReal y) {
    PetscReal sigma = 0.3;
    PetscReal x0 = 0.65, y0 = 0.65;
    PetscReal amp = 1.0;
    PetscReal r2 = (x-x0)*(x-x0) + (y-y0)*(y-y0);
    PetscReal expterm = PetscExpReal( - r2 / (sigma*sigma) );
    return amp * expterm * 4.0/(sigma * sigma) * ( r2/(sigma*sigma) - 1.0 );
    //return 2.0 * ( (1.0 - 6.0*x*x) * y*y * (1.0 - y*y) + (1.0 - 6.0*y*y) * x*x * (1.0 - x*x) );
}

//STARTMATRIX
PetscErrorCode formMatrix(DM da, Mat A) {
    DMDALocalInfo  info;
    MatStencil     row, col[5];
    PetscReal      hx, hy, v[5];
    PetscInt       i, j, ncols;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    for (j = info.ys; j < info.ys+info.ym; j++) {
        for (i = info.xs; i < info.xs+info.xm; i++) {
            row.j = j;           // row of A corresponding to (x_i,y_j)
            row.i = i;
            col[0].j = j;        // diagonal entry
            col[0].i = i;
            ncols = 1;
            if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
                v[0] = 1.0;      // on boundary: trivial equation
            } else {
                v[0] = 2.*(hy/hx + hx/hy); // interior: build a row
                if (i-1 > 0) {
                    col[ncols].j = j;    col[ncols].i = i-1;
                    v[ncols++] = -hy/hx;
                }
                if (i+1 < info.mx-1) {
                    col[ncols].j = j;    col[ncols].i = i+1;
                    v[ncols++] = -hy/hx;
                }
                if (j-1 > 0) {
                    col[ncols].j = j-1;  col[ncols].i = i;
                    v[ncols++] = -hx/hy;
                }
                if (j+1 < info.my-1) {
                    col[ncols].j = j+1;  col[ncols].i = i;
                    v[ncols++] = -hx/hy;
                }
            }
            PetscCall(MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    return 0;
}
//ENDMATRIX

//STARTEXACT
PetscErrorCode formExact(DM da, Vec uexact) {
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **auexact;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    PetscCall(DMDAVecGetArray(da, uexact, &auexact));
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
PetscErrorCode formRHS(DM da, Vec b) {
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **ab;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    PetscCall(DMDAVecGetArray(da, b, &ab));
    for (j = info.ys; j < info.ys + info.ym; j++)
    {
        y = j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++)
        {
            x = i * hx;
            if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1)
            {
                ab[j][i] = ufunction(x, y); // on boundary: 1*u = uexact(x,y)
            }
            else
            {
                ab[j][i] = -hx * hy * d2ufunction(x, y); // interior: f(x_i,y_j)
                // // lift dirichlet BCs into interior equations
                if (i == 1)
                {
                    ab[j][i] += hy / hx * ufunction(x - hx, y);
                }
                if (i == info.mx - 2)
                {
                    ab[j][i] += hy / hx * ufunction(x + hx, y);
                }
                if (j == 1)
                {
                    ab[j][i] += hx / hy * ufunction(x, y - hy);
                }
                if (j == info.my - 2)
                {
                    ab[j][i] += hx / hy * ufunction(x, y + hy);
                }
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(da, b, &ab));
    return 0;
}
//ENDRHS

//STARTf
PetscErrorCode  formF(DM da, Vec f) {
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **af;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    PetscCall(DMDAVecGetArray(da, f, &af));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = i * hx;
            af[j][i] =  -d2ufunction(x,y);  // interior: f(x_i,y_j)
        }
    }
    PetscCall(DMDAVecRestoreArray(da, f, &af));
    return 0;
}
//ENDRHS

//STARTRANKMAP
PetscErrorCode formRankMap(DM da, Vec rankmap, PetscInt rank) {
    PetscInt       i, j;
    PetscReal      **ab;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(DMDAVecGetArray(da, rankmap, &ab));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        for (i=info.xs; i<info.xs+info.xm; i++) {
            ab[j][i] = (PetscReal) rank;
        }
    }
    PetscCall(DMDAVecRestoreArray(da, rankmap, &ab));
    return 0;
}
//ENDRANKMAP
