/*
  1-D advection_diffusion with IMEX (ARKIMEX) using DMDA.

  PDE (periodic):
      u_t = nu * u_xx  +  (-a * u_x)

      u(t=0) = exp(-(x-x0)^2 / (2*sigma^2)) 

  Split:
    - Explicit part (RHSFunction):   G(u) =  -a u_x
    - Implicit part (IFunction):    F(t,u,udot) = udot - nu u_xx

  Spatial discretization: 2nd-order centered differences on a uniform grid.

  Build:
    make ts_advection_diffusion_dmda



  Run:
    ./ts_advection_diffusion_dmda -ts_type arkimex -ts_max_time 1 -ts_dt 1e-3 -ts_monitor
    ./ts_advection_diffusion_dmda -da_grid_x 400 -a 2.0 -nu 1e-2 -ts_type arkimex
*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>

typedef enum {CENTERED, UPWIND, THIRDORDER, VANLEER, KOREN} AdvectionScheme;
static const char* AdvectionSchemes[] = {"centered","upwind","thirdorder","vanleer","koren",
                                     "AdvectionScheme", "", NULL};


typedef struct {
  DM        da;
  PetscReal L, dx;
  PetscReal a;   /* convection speed */
  PetscReal nu;  /* diffusion coeff */
  PetscReal x0, sigma; /* IC parameters, not used in this example but could be set via options */
  AdvectionScheme scheme; /* type of advection scheme */
} AppCtx;

/* van Leer (1974) limiter is formula (1.11) in section III.1 of
Hundsdorfer & Verwer (2003) */
static PetscReal vanleer(PetscReal theta) {
    const PetscReal abstheta = PetscAbsReal(theta);
    return 0.5 * (theta + abstheta) / (1.0 + abstheta);
}

/* Koren (1993) limiter is formula (1.7) in section III.1 of
Hundsdorfer & Verwer (2003) */
static PetscReal koren(PetscReal theta) {
    const PetscReal z = (1.0/3.0) + (1.0/6.0) * theta;
    return PetscMax(0.0, PetscMin(1.0, PetscMin(z, theta)));
}

static PetscReal theta(PetscReal *u, PetscInt i) {
  const PetscReal epsilon = 1e-10; /* small number to prevent division by zero */
  return (u[i] - u[i-1] + epsilon)/(u[i+1] -u[i] + epsilon);
}

// Routines to set exact solution and initial conditions for a Gaussian pulse advecting with speed a and diffusing with diffusivity nu, in a periodic domain of length L.

// return analytic solution for periodic advecting Gaussian pulses with diffusion
static PetscScalar DiffusingGaussian(PetscReal t, PetscReal x, void *ctx)
{
  AppCtx *app = (AppCtx*)ctx;
  PetscReal sum = 0.0;
  PetscInt n;
  PetscReal x0 = app->x0, sigma = app->sigma, a = app->a, nu = app->nu, L = app->L;
  /* Variance grows due to diffusion: sigma^2(t) = sigma^2(0) + 2*nu*t */
  PetscReal variance = sigma*sigma + 2.0*nu*t;
  PetscReal normalization = sigma  / PetscSqrtReal(variance);
  
  /* Sum over periodic images: typically need only a few terms for good accuracy */
  for (n = -3; n <= 3; n++) {
    PetscReal x_shifted = x - x0 - a*t + n*L; /* periodic shift */
    sum += PetscExpReal(-0.5 * PetscPowReal(x_shifted, 2) / variance);
  }
  
  return normalization * sum;
}


static PetscErrorCode SetInitialCondition(Vec U, AppCtx *app)
{
  DM            da = app->da;
  DMDALocalInfo info;
  PetscScalar  *u;
  PetscInt      i;
  PetscReal     dx = app->dx;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAVecGetArray(da,U,&u));
  for (i = info.xs; i < info.xs + info.xm; i++) {
    PetscReal x = i*dx;
    u[i] = DiffusingGaussian(0.0, x, (void*)app);
  }
  PetscCall(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// set exact solution for advecting Gaussian pulse with speed a, with diffusion
static PetscErrorCode SetExactSolution(Vec U, PetscReal t, void *ctx)
{
  AppCtx        *app = (AppCtx*)ctx;  
  DM            da = app->da;
  DMDALocalInfo info;
  PetscScalar  *u;
  PetscInt      i;
  PetscReal     dx = app->dx;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAVecGetArray(da,U,&u));
  for (i = info.xs; i < info.xs + info.xm; i++) {
    PetscReal x = i*dx;
    u[i] = DiffusingGaussian(t, x, (void*)app);
  }
  PetscCall(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Routines to compute the RHS and IFunction residual for the IMEX split, 
// and the Jacobian of the IFunction for the implicit solve. 
// The RHS is just the convection term, while the IFunction includes the time derivative and diffusion term. 
// The Jacobian is tridiagonal due to the second derivative in diffusion.

/* Explicit convection: G(u) = -a u_x */
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec R, void *ctx)
{
  AppCtx            *app = (AppCtx*)ctx;
  DM                 da  = app->da;
  DMDALocalInfo      info;
  Vec                Uloc;
  PetscScalar *u;   /* local array including ghosts */
  PetscScalar       *r;   /* global array (owned part only) */
  PetscInt           i;
  const PetscReal    a  = app->a, dx = app->dx;
  PetscReal         a_pos = PetscMax(a,0.0), a_neg = PetscMin(a,0.0); 
  PetscReal         f_pos, f_neg;
  PetscReal         th, thp, psi, psip;
  AdvectionScheme     scheme = app->scheme;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da,&info));

  PetscCall(DMGetLocalVector(da,&Uloc));
  PetscCall(DMGlobalToLocalBegin(da,U,INSERT_VALUES,Uloc));
  PetscCall(DMGlobalToLocalEnd(da,U,INSERT_VALUES,Uloc));

  /* IMPORTANT: use DMDAVecGetArrayRead on the LOCAL vector */
  PetscCall(DMDAVecGetArrayRead(da,Uloc,&u));
  PetscCall(DMDAVecGetArray(da,R,&r));

  if (scheme == CENTERED) {
    for (i = info.xs; i < info.xs + info.xm; i++) {
      /* centered difference; neighbors are valid via ghost points, including periodic wrap */
      r[i] = -a * (u[i+1] - u[i-1]) / (2.0*dx);
    }
  }
  else if (scheme == UPWIND) {
    i = info.xs;
    f_neg = a_pos * u[i-1] + a_neg * u[i]; /* flux at i-1/2 */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      /* upwind difference: use backward difference if a>0, forward if a<0 */
        f_pos = a_pos * u[i] + a_neg * u[i+1]; /* flux at i+1/2 */
        r[i] =  -(f_pos - f_neg) / dx;
        f_neg = f_pos; /* shift for next iteration */
    }
  }
  else if (scheme == THIRDORDER) {
    i = info.xs;
    f_neg = a_pos * (   -u[i-2] + 5.0*u[i-1] + 2.0*u[i]) / 6.0 +
            a_neg * (2.0*u[i-1] + 5.0*u[i]   -     u[i+1]) / 6.0; /* flux at i-1/2 */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      /* third-order upwind-biased difference */
        f_pos = a_pos * (   -u[i-1] + 5.0*u[i] + 2.0*u[i+1]) / 6.0 +
                a_neg * (2.0*u[i]   + 5.0*u[i+1] -     u[i+2]) / 6.0; /* flux at i+1/2 */
        r[i] =  -(f_pos - f_neg) / dx;
        f_neg = f_pos; /* shift for next iteration */ 
    }
  } 
  else if (scheme == KOREN) {
    i = info.xs;
    th = theta( u , i-1);
    thp = theta( u, i);
    psi = koren(th);
    psip = koren(1./thp);
    f_neg = a_pos * ((1-psi)*u[i-1] + psi*u[i] )+
           a_neg *  ((1-psip)*u[i] + psip*u[i-1]); /* flux at i-1/2 */
    for (i = info.xs; i < info.xs + info.xm; i++) {
        th = thp;
        thp = theta(u, i+1);
        psi = koren(th);
        psip = koren(1./thp);
        f_pos = a_pos * ((1-psi)*u[i] + psi*u[i+1] )+
                a_neg *  ((1-psip)*u[i+1] + psip*u[i]);  /* flux at i+1/2 */
        r[i] =  -(f_pos - f_neg) / dx;
        f_neg = f_pos; /* shift for next iteration */
    }
  }
  else if (scheme == VANLEER) {
    i = info.xs;
    th = theta( u , i-1);
    thp = theta( u, i);
    psi = vanleer(th);
    psip = vanleer(1./thp);
    f_neg = a_pos * ((1-psi)*u[i-1] + psi*u[i] )+
           a_neg *  ((1-psip)*u[i] + psip*u[i-1]); /* flux at i-1/2 */
    for (i = info.xs; i < info.xs + info.xm; i++) {
        th = thp;
        thp = theta(u, i+1);
        psi = vanleer(th);
        psip = vanleer(1./thp);
        f_pos = a_pos * ((1-psi)*u[i] + psi*u[i+1] )+
                a_neg *  ((1-psip)*u[i+1] + psip*u[i]);  /* flux at i+1/2 */
        r[i] =  -(f_pos - f_neg) / dx;
        f_neg = f_pos; /* shift for next iteration */
    }
  }
  else {
    /* default to centered if invalid scheme */
    for (i = info.xs; i < info.xs + info.xm; i++) {
      r[i] = -a * (u[i+1] - u[i-1]) / (2.0*dx);
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(da,Uloc,&u));
  PetscCall(DMDAVecRestoreArray(da,R,&r));
  PetscCall(DMRestoreLocalVector(da,&Uloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Implicit diffusion residual: F = udot - nu u_xx */
static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
  AppCtx            *app = (AppCtx*)ctx;
  DM                 da  = app->da;
  DMDALocalInfo      info;
  Vec                Uloc;
  const PetscScalar *u;    /* local array including ghosts */
  const PetscScalar *udot; /* global array */
  PetscScalar       *f;    /* global array */
  PetscInt           i;
  const PetscReal    nu = app->nu, dx = app->dx;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da,&info));

  PetscCall(DMGetLocalVector(da,&Uloc));
  PetscCall(DMGlobalToLocalBegin(da,U,INSERT_VALUES,Uloc));
  PetscCall(DMGlobalToLocalEnd(da,U,INSERT_VALUES,Uloc));

  PetscCall(DMDAVecGetArrayRead(da,Uloc,&u));
  PetscCall(DMDAVecGetArrayRead(da,Udot,&udot));
  PetscCall(DMDAVecGetArray(da,F,&f));

  for (i = info.xs; i < info.xs + info.xm; i++) {
    PetscScalar uxx = (u[i+1] - 2.0*u[i] + u[i-1])/(dx*dx);
    f[i] = udot[i] - nu*uxx;
  }

  PetscCall(DMDAVecRestoreArrayRead(da,Uloc,&u));
  PetscCall(DMDAVecRestoreArrayRead(da,Udot,&udot));
  PetscCall(DMDAVecRestoreArray(da,F,&f));
  PetscCall(DMRestoreLocalVector(da,&Uloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Jacobian of IFunction: J = a_shift*I - nu*Dxx */
static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal a_shift,
                               Mat J, Mat P, void *ctx)
{
  AppCtx        *app = (AppCtx*)ctx;
  DM             da  = app->da;
  DMDALocalInfo  info;
  PetscInt       i;
  PetscReal      nu = app->nu, dx = app->dx;
  PetscScalar    val[3];
  MatStencil     row, col[3];

  PetscFunctionBeginUser;
  PetscCall(DMDAGetLocalInfo(da,&info));

  PetscCall(MatZeroEntries(P));
  for (i = info.xs; i < info.xs + info.xm; i++) {
    row.i = i;

    col[0].i = i-1; val[0] = -nu/(dx*dx);
    col[1].i = i;   val[1] = a_shift + 2.0*nu/(dx*dx);
    col[2].i = i+1; val[2] = -nu/(dx*dx);

    PetscCall(MatSetValuesStencil(P,1,&row,3,col,val,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}



int main(int argc, char **argv)
{
  TS      ts;
  Vec     U, Uexact;
  Mat     J;
  AppCtx  app;
  PetscInt N = 200;
  PetscReal error, normu = 0.0;
  PetscInt stencil_width = 1; /* default for centered differences; use 1 for upwind, 2 for third-order */

  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  app.L  = 1.0;
  app.a  = 1.0;
  app.nu = 1e-2;
  app.x0 = 0.3; /* default center of initial condition */
  app.sigma = 0.1; /* default width of initial condition */
  app.scheme = CENTERED; /* default advection scheme */


  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-L",&app.L,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-a",&app.a,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-nu",&app.nu,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-x0",&app.x0,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-sigma",&app.sigma,NULL));
  PetscCall(PetscOptionsGetEnum(NULL,NULL,"-scheme", AdvectionSchemes, (PetscEnum*)&app.scheme, NULL));
  
  // determine stencil width based on advection scheme
  stencil_width = 1; /* default for everyone but THIRDORDER */
  if (app.scheme == THIRDORDER) {
    stencil_width = 2; /* need two neighbors on each side for third-order upwind-biased */
  }
  /* 1D DMDA with 1 dof/node, stencil width 1.
     Periodic boundary so centered differences are valid everywhere. */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, N,
                         1, /* dof */
                         stencil_width, /* stencil width for centered differences; use 1 for upwind, 2 for third-order */
                         NULL, &app.da));
  PetscCall(DMSetFromOptions(app.da));
  PetscCall(DMSetUp(app.da));

  // set coordinates for da
  PetscCall(DMDASetUniformCoordinates(app.da,0.0,app.L,0.0,0.0,0.0,0.0));

  /* Get actual global size after -da_grid_x etc. */
  PetscInt M;
  PetscCall(DMDAGetInfo(app.da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));
  app.dx = app.L / (PetscReal)M;

  PetscCall(DMCreateGlobalVector(app.da,&U));
  PetscCall(PetscObjectSetName((PetscObject)U,"Solution"));
  PetscCall(SetInitialCondition(U,&app));

  PetscCall(DMCreateGlobalVector(app.da,&Uexact));
  PetscCall(PetscObjectSetName((PetscObject)Uexact,"Exact Solution"));
  PetscCall(SetExactSolution(Uexact,0.0,&app));

  /* Create matrix compatible with DMDA stencil */
  PetscCall(DMCreateMatrix(app.da,&J));

  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetDM(ts,app.da));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));

  /* IMEX split */
  PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction,&app));
  PetscCall(TSSetIFunction(ts,NULL,IFunction,&app));
  PetscCall(TSSetIJacobian(ts,J,J,IJacobian,&app));

  PetscCall(TSSetType(ts,TSARKIMEX));
  PetscCall(TSSetTime(ts,0.0));
  PetscCall(TSSetMaxTime(ts,1.0));
  PetscCall(TSSetTimeStep(ts,1e-3));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));

  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts,U));

  PetscCall(SetExactSolution(Uexact,1.0,&app));
  PetscCall(VecNorm(Uexact,NORM_2,&normu));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of exact solution: %g\n",(double)normu));
  PetscCall(VecAXPY(Uexact,-1.0,U));
  PetscCall(VecNorm(Uexact,NORM_2,&error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"relative Error in solution: %g\n",(double)error/(double)normu));

  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&app.da));
  PetscCall(PetscFinalize());
  return 0;
}
