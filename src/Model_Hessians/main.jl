# =============================================================================
# src/Model_Hessians/main.jl
#
# Axis 2: model Hessians.
#
# Must be included before `Subproblem/` and `Trust-region/`: both dispatch on
# `ModelHessian`, so the type has to exist first.
# =============================================================================

include("model_hessian.jl")
