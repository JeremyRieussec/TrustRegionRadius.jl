# =============================================================================
# src/Subproblem/main.jl
#
# Axis 3: trust-region subproblem solvers.
#
# Requires `ModelHessian` (from Model_Hessians/) -- `solve_subproblem!` takes
# the model as an argument so one set of solvers serves every model.
# =============================================================================

include("subproblem.jl")
