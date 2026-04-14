# This example is based on
# https://fenicsproject.org/docs/dolfin/latest/python/demos/cahn-hilliard/demo_cahn-hilliard.py.html
# and
# https://github.com/firedrakeproject/firedrake-bench/blob/master/cahn_hilliard/firedrake_cahn_hilliard.py

import firedrake_ts
from firedrake import *
import numpy as np
import json 

# Model parameters
lmbda = 1.0e-02  # surface parameter
dt = 5.0e-06  # time step

N = 4
levels = 4  
N_max = N * 2 ** levels
coarsemesh = UnitSquareMesh(N, N, quadrilateral=True)
hierarchy = MeshHierarchy(coarsemesh, levels)
mesh = hierarchy[-1]

family = "CG"
degree = 1  
V = FunctionSpace(mesh, family, degree)
ME = V * V

# Define functions
u = Function(ME)  # current solution
u_dot = Function(ME)  # time derivative

# Split mixed functions
c, mu = split(u)
c_dot, mu_dot = split(u_dot)

# name the subfunctions for output
u.subfunctions[0].rename("Concentration")
u.subfunctions[1].rename("Chemical_Potential")

# Define test functions
c_t, mu_t = TestFunctions(ME)


# Compute the chemical potential df/dc
c = variable(c)
f = 100 * c ** 2 * (1 - c) ** 2
dfdc = diff(f, c)

# Weak form of the residual equations
Fc = c_t * c_dot  * dx + dot(grad(mu), grad(c_t)) * dx
Fmu = mu_t * mu * dx - mu_t * dfdc * dx - lmbda * dot(grad(mu_t), grad(c)) * dx
F = Fc + Fmu

rng = np.random.default_rng(11)
c , mu = u.subfunctions
with c.dat.vec as v:
    v[:]=0.63 + 0.2*(0.5-rng.random(v.size))

t_max = 0.001
params = {
    "ts_type": "bdf",
    "ts_bdf_order": 3,
    "ts_equation_type": "implicit",
    "ts_dt": dt,
    "ts_adapt_dt_min": 1.e-6,
    "ts_max_time": t_max,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_linesearch_type": "basic",
    "snes_linesearch_max_it": 1,
    "ts_monitor": None,
    "snes_monitor": None,
    "snes_rtol": 1e-9,
}


outfile = VTKFile("result/cahn-hilliard.pvd")
outfile.write(u.subfunctions[0],u.subfunctions[1], time=0.0)


def ts_monitor(ts, steps, time, X):
    outfile.write(u.subfunctions[0],u.subfunctions[1], time=time)


problem = firedrake_ts.DAEProblem(F, u, u_dot, (0.0, t_max))
solver = firedrake_ts.DAESolver(problem, solver_parameters=params, monitor_callback=ts_monitor)

solver.solve()

print("Solving Cahn-Hilliard equation on {}x{} Q{} mesh with parameters = {}".
      format(N_max, N_max, degree, json.dumps(params, indent=4)))
