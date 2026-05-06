# Simple Helmholtz equation but cast as a nonlinear problem
# =========================
#
# Let's start by considering the modified Helmholtz equation on a unit square,
# :math:`\Omega`, with boundary :math:`\Gamma`:
#
# .. math::
#
#    -\nabla^2 u + u^n &= f
#
#    \nabla u \cdot \vec{n} &= 0 \quad \textrm{on}\ \Gamma
#
# for some known function :math:`f`. The solution to this equation will
# be some function :math:`u\in V`, for some suitable function space
# :math:`V`, that satisfies these equations. Note that this is the
# Helmholtz equation that appears in meteorology, rather than the
# indefinite Helmholtz equation :math:`\nabla^2 u + u = f` that arises
# in wave problems.
#
# We transform the equation into weak form by multiplying by an arbitrary
# test function in :math:`V`, integrating over the domain and then
# integrating by parts. The variational problem so derived reads: find
# :math:`u \in V` such that:
#
# .. math::
#
#    \require{cancel}
#    \int_\Omega \nabla u\cdot\nabla v  + uv\ \mathrm{d}x = \int_\Omega
#    vf\ \mathrm{d}x + \cancel{\int_\Gamma v \nabla u \cdot \vec{n} \mathrm{d}s}
#
# Here we assume homogeneous Dirichlet boundary conditions, 
# 
# 
# We can choose the function :math:`f`, so we take:
#
# .. math::
#
#    f = (1.0 + 8.0\pi^2)\sin(2\pi x)\sin(2\pi y)
#
# which conveniently yields the analytic solution:
#
# .. math::
#
#    u = \sin(2\pi x)\sin(2\pi y)
#
# However we wish to employ this as an example for the finite element
# method, so lets go ahead and produce a numerical solution.
#
# First, we always need a mesh. Let's have a multi-level quadrilateral element unit square::

import json
from firedrake import *

# set up mesh hieararchy for geometric multigrid
 
N = 4
levels = 4
coarse_mesh = UnitSquareMesh(N, N, quadrilateral=True)    
hierarchy = MeshHierarchy(coarse_mesh, levels) 
mesh = hierarchy[-1]
N_fine = N * 2**levels 

# We need to decide on the function space in which we'd like to solve the
# problem. Let's use piecewise  Qp linear functions continuous between
# elements::
p = 2
V = FunctionSpace(mesh, "CG", p)

# We'll also need the test and trial functions corresponding to this
# function space::

u = Function(V, name="u")
v = TestFunction(V)

# We declare a function over our function space and give it the
# value of our right hand side function::

f = Function(V)
x, y = SpatialCoordinate(mesh)

# exponent in non-linear term
n = 4
f.interpolate((8*pi*pi)*sin(x*pi*2)*sin(y*pi*2) + (sin(x*pi*2) * sin(y*pi*2))**n)

# We can now define the weak form of the residual function F(u)::
F = (inner(grad(u), grad(v)) + inner(u**n, v)) * dx - inner(f, v) * dx

# For this solution we will need to implement dirichlet boundary conditions, so we set the value of the solution to be zero on the boundary of the domain::

bc = DirichletBC(V, 0, "on_boundary")

# set the petsc solver parameters to use the nonlinear solver SNES with a direct LU factorisation preconditioner. We also set some additional parameters to monitor the convergence of the nonlinear and linear solvers and to set the relative tolerance for convergence. For more details on how to specify solver parameters, see the section of the manual on :doc:`solving PDEs <../solving-interface>`.

parameters = {'ksp_type': 'cg', 
              'pc_type': 'mg',
              'pc_factor_mat_solver_type': 'mumps',
              'snes_type': 'newtonls',
              'snes_monitor': None,
              'snes_converged_reason': None,
              'snes_rtol': 1e-8,
              'snes_atol': 1e-10,
              'ksp_monitor': None,
              'ksp_converged_reason': None,
              'ksp_rtol': 1e-6,
              'ksp_atol': 1e-10}

print("\nSolving {}x{} non-Linear system in Q{} with parameters: {}".
      format(N_fine, N_fine, p, json.dumps(parameters, indent=4)))

solve(F == 0, u, solver_parameters=parameters, bcs=bc)

# Print the absolute and relative L2 error of the solution compared to the known analytic solution.
#  We can compute the L2 error by integrating the square of the difference between the numerical and analytic solutions over the domain, and then taking the square root. The relative L2 error is computed by dividing the absolute L2 error by the L2 norm of the analytic solution.

f.interpolate(sin(x*pi*2)*sin(y*pi*2))
abs_error = sqrt(assemble(dot(u - f, u - f) * dx))
rel_error = abs_error / sqrt(assemble(dot(f, f) * dx))
print(f'\nL2 error: {abs_error}')
print(f'Relative L2 error: {rel_error}')


# Next, we might want to look at the result, so we output our solution
# to a file::

VTKFile("helmholtz_snes.pvd").write(u)

