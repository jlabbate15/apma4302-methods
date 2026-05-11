# Solve the biharmonic equation as a coupled system of two Poisson equations.
# This code reproduces the manufactured solution for the biharmonic equation from the Finite Difference code in C,
#  but using the Firedrake finite element library.
import firedrake_ts
from firedrake import *

nu_results_file = "result/convection_hw4_q4_nu_vs_mesh.txt"
with open(nu_results_file, "w") as f:
    f.write("levels Nfine Ra Nu\n")

for levels in range(3,7):
    N = 2 
    # levels = 5
    Nfine = N*2**levels
    base_mesh = UnitSquareMesh(N, N, quadrilateral=True) # Q mesh
    Hierarchy = MeshHierarchy(base_mesh, levels)
    mesh = Hierarchy[-1]

    # set RHS function and True solutions
    # f = Function(V, name="rhs")
    # set initial conditions
    x, z = SpatialCoordinate(mesh)

    # Second, solve for the temperature field T(t) with the flow field (ω,ψ) as the initial condition.
    Ra = 10**4

    V = FunctionSpace(mesh, "Lagrange", 1) # Q1 (linearly interpolated mesh points), function space for scalers
    ME = MixedFunctionSpace([V, V, V], name=["vorticity", "streamfunction", "temperature"])
    # Define functions
    u = Function(ME)  # current solution
    u.subfunctions[0].rename("vorticity")
    u.subfunctions[1].rename("streamfunction")
    u.subfunctions[2].rename("temperature")
    # Split mixed functions
    omega, psi, T = split(u)
    # Define the time derivative of the solution function
    udot = Function(ME)
    omegadot, psidot, Tdot = split(udot)
    # Define test functions for omega, psi, and T
    omega_t, psi_t, T_t = TestFunctions(ME)

    v = curl(psi)


    # First, solve for the flow field (ω,ψ) with ω = ψ = 0 on ∂Ω using the initial temperature field T(t=0) = (1-y) + 0.1*cos(pi*x).
    A = 0.1
    T_ic = Function(V, name="Temperature_IC")
    T_ic.interpolate((1.0 - z) + A * cos(pi * x))
    f_ic = T_ic.dx(0)

    ME_flow = MixedFunctionSpace([V, V])
    u_flow = Function(ME_flow)
    omega0, psi0 = split(u_flow)
    omega_t0, psi_t0 = TestFunctions(ME_flow)
    Fomega_ic = inner(grad(omega_t0), grad(omega0)) * dx - omega_t0 * f_ic * dx
    Fpsi_ic = inner(grad(psi_t0), grad(psi0)) * dx - psi_t0 * omega0 * dx
    F_flow = Fomega_ic + Fpsi_ic
    bcs_flow = [
        DirichletBC(ME_flow.sub(0), 0., "on_boundary"),
        DirichletBC(ME_flow.sub(1), 0., "on_boundary"),
    ]
    params_general = {
        "snes_type": "ksponly",
        "ksp_rtol": 1e-6,
        "ksp_atol": 1e-10,
        "ksp_monitor": None,
        "log_view": None,
    }
    params = {
        "ksp_type": "fgmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "multiplicative",
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "mg",
        # "fieldsplit_0_pc_mg_levels": 8,
        "fieldsplit_0_pc_mg_galerkin": "none",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "mg",
        # "fieldsplit_1_pc_mg_levels": 8,
        "fieldsplit_1_pc_mg_galerkin": "none",
    }
    params.update(params_general)
    problem = NonlinearVariationalProblem(F_flow, u_flow, bcs=bcs_flow)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    u.subfunctions[0].assign(u_flow.subfunctions[0])
    u.subfunctions[1].assign(u_flow.subfunctions[1])
    u.subfunctions[2].assign(T_ic)


    # Now time-dependent solver
    # Weak statement of the equations, after integrating by parts
    FT = (T_t * Tdot + T_t * inner( v, grad(T) ) + (1/Ra)*inner( grad(T_t), (grad(T)) ))*dx
    Fomega = inner( grad(omega_t), (grad(omega)) ) * dx - T.dx(0) * omega_t * dx
    Fpsi = inner( grad(psi_t), (grad(psi)) ) * dx -  psi_t * omega * dx
    F = FT + Fomega + Fpsi

    # boundary conditions
    # T = 1 at the bottom (y=0) and T = 0 at the top (y=1)
    bc_temp_bot = DirichletBC(ME.sub(2), Constant(1.0), 3) # bottom
    bc_temp_top = DirichletBC(ME.sub(2), Constant(0.0), 4) # top

    # psi = 0 and omega = 0 on ALL boundaries (on dOmega)
    bc_psi_all   = DirichletBC(ME.sub(1), Constant(0.0), "on_boundary")
    bc_omega_all = DirichletBC(ME.sub(0), Constant(0.0), "on_boundary")

    bcs = [bc_temp_bot, bc_temp_top, bc_psi_all, bc_omega_all]

    # Neumann boundary conditions on left and right boundaries do not need to be specified

    t_init = 0.0
    t_max = 10**5
    params = {'ts_type': 'bdf',
            'ts_bdf_order': 2,
            'ts_dt': 0.1, # first timestep dt, then adaptative
            'ts_monitor': None,
            'ts_rtol': 1e-6,
            'ts_atol': 1e-10,
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'ts_max_time': t_max,
            'ts_adapt_dt_min': 1.e-4,
            'ts_exact_final_time': "matchstep",
            'ts_time': t_init,
            'ts_max_time': t_max
        }
    udot.assign(0.0)  # otherwise is random

    # Replace NonlinearVariationalProblem with DAEProblem
    tspan = (t_init, t_max)
    problem = firedrake_ts.DAEProblem(F, u, udot, tspan, bcs=bcs)

    Vv = VectorFunctionSpace(mesh, "Lagrange", 1)
    v_output = Function(Vv, name="Velocity")
    v_output.interpolate(curl(u.subfunctions[1]))

    outfile = VTKFile("result/convection_hw4_q4_Ra"+str(Ra)+"mesh"+str(Nfine)+".pvd")
    outfile.write(u.subfunctions[0], u.subfunctions[1], u.subfunctions[2], v_output) # vorticity, streamfunction, temperature, velocity

    def monitor(ts, step, t, x_vec):
        v_output.interpolate(curl(u.subfunctions[1]))
        outfile.write(u.subfunctions[0], u.subfunctions[1], u.subfunctions[2], v_output, time=t)

    # Replace NonlinearVariationalSolver with DAESolver
    solver = firedrake_ts.DAESolver(problem, solver_parameters=params, monitor_callback=monitor)
    solver.solve()

    # Calculate the Nusselt Number
    num = assemble(u.subfunctions[2].dx(1) * ds(4)) # Integral at Top
    den = assemble(u.subfunctions[2].dx(1) * ds(3)) # Integral at Bottom
    # print(f"DEBUG: Top Integral = {num}, Bottom Integral = {den}")
    Nu = -(num / den)
    # print(f"Final Nusselt Number: {Nu}")
    with open(nu_results_file, "a") as f:
        f.write(f"{levels} {Nfine} {Ra} {Nu}\n")