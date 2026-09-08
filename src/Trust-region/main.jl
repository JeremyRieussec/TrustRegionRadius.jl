# =============================================================================
# src/Trust-region/main.jl
#
# The three solvers, one per problem class.
#
#   common.jl         TRParams, TRCore, TRState, TRTrace, and the one iteration
#                     `_tr_step!` that all three share
#   deterministic.jl  DeterministicTRSolver
#   expectation.jl    ExpectationTRSolver   (also _confirm_stop, _attach_sampling!)
#   finitesum.jl      FiniteSumTRSolver
#   entry.jl          tr_solve, dispatching on the oracle
#
# expectation.jl before finitesum.jl: the finite-sum loop calls _confirm_stop and
# _attach_sampling!, which are defined there rather than duplicated.
# =============================================================================

include("common.jl")
include("deterministic.jl")
include("expectation.jl")
include("finitesum.jl")
include("entry.jl")