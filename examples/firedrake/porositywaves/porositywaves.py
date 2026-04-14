# This example is based on
# TerraFERMA solitary wave benchmark
import firedrake_ts
from firedrake import *
import numpy as np

# Model parameters
h_on_delta = 64.0
h_on_delta = 100.0
hsquared = h_on_delta ** 2  # h/delta (system size on compaction length)^2
A = 3.  # amplitude of porosity perturbation
sigma = 0.1  # width of porosity perturbation

mesh = PeriodicRectangleMesh(50, 100, 0.5, 1.0, quadrilateral=True, direction="both")
V = FunctionSpace(mesh, "Lagrange", 2)
ME = V * V

# Define test functions for porosity and pressure
f_t, p_t = TestFunctions(ME)

# Define functions
u = Function(ME)  # current solution
u_dot = Function(ME)  # time derivative

# Split mixed functions
u.subfunctions[0].rename = "Porosity"   
u.subfunctions[1].rename = "Pressure"

f, p = split(u)
f_dot, p_dot = split(u_dot)

# set permeability and bulk viscosity parameters
K = f ** 3
Xi = hsquared
ghat = as_vector((0, -1))

# Weak statement of the equations
Ff = f_t * f_dot * dx -  f_t * Xi * p * dx
Fp = inner( grad(p_t), K * (grad(p) + ghat) ) * dx + Xi * p_t * p * dx
F = Ff + Fp     


# set initial conditions
x, y = SpatialCoordinate(mesh)
f_init = 1. + A * exp(-((x - 0.25) ** 2 + (y - 0.25) ** 2) / (2 * sigma ** 2))
u.subfunctions[0].interpolate(f_init)
u.subfunctions[1].interpolate(0)    

    
params = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_rtol": 1e-9,
    "snes_atol": 1e-10,
    "snes_stol": 1e-16,
    "ksp_rtol": 1e-6,
    "ksp_atol": 1e-10,
    "snes_monitor": None,
    "ts_monitor": None,
    "ts_type": "bdf",
    "ts_view": None,
    "ts_max_time": .5,
    "ts_adapt_dt_min": 1.e-6
}


outfile = VTKFile("result/porosity-waves.pvd")
outfile.write(u.subfunctions[0], u.subfunctions[1], names=["Porosity", "Pressure"], time=0.0,)


def ts_monitor(ts, steps, time, X):
    outfile.write(u.subfunctions[0], u.subfunctions[1], names=["Porosity", "Pressure"], time=time)


problem = firedrake_ts.DAEProblem(F, u, u_dot, (0.0, 2.0))
solver = firedrake_ts.DAESolver(problem, solver_parameters=params, monitor_callback=ts_monitor)

dt = 1.e-3  # time step

ts = solver.ts
ts.setTimeStep(dt)
ts.setEquationType(PETSc.TS.EquationType.IMPLICIT)
ts.setFromOptions()

solver.solve()
