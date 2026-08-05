module TrustRegionRadius

using LinearAlgebra
using Random
using Printf

using NLPModels
using SolverCore
using Krylov
using LinearOperators

import SolverCore: solve!, reset!
import LinearAlgebra: mul!, issymmetric, ishermitian

# =============================================================================
# Includes.
#
# `Problems` first: the class hierarchy is what the model Hessians, the sampling
# rules and the three solvers are all checked against, so nothing can be defined
# before it.
#
# Then the four axes, then the solvers, then the benchmark layer.
# =============================================================================

include("Problems/main.jl")           # DeterministicProblem / Expectation / FiniteSum
include("Radius_updates/main.jl")     # RadiusRule and every mechanism
include("Model_Hessians/main.jl")     # ModelHessian and every model
include("Subproblem/main.jl")         # SubproblemSolver and every subsolver
include("Second_order/main.jl")       # τ-anchoring, curvature, EigenPoint
include("Sampling/main.jl")           # sampled problems, sampling rules, oracles
include("Likelihood/main.jl")         # BHHH, BHHH-2, Gauss-Newton; likelihoods
include("Trust-region/main.jl")       # TRParams and the three solvers
include("Benchmark/main.jl")          # profiles and the run matrix

# =============================================================================
# Exports
# =============================================================================

# ---- Problem classes --------------------------------------------------------
export AbstractProblem, DeterministicProblem, SampledProblem
export ExpectationProblem, FiniteSumProblem, ScoredProblem, LikelihoodProblem
export NLSProblem
export problem_class, population, n_terms, has_scores, has_truth, full_batch
export underlying_problem, required_problem, check_model_problem
export check_rule_problem, check_population_cap, user_cap

# ---- Axis 1: radius mechanisms ----------------------------------------------
export RadiusRule
export RDelta, RStep, RDFO, RGrad, RGradCapped
export RAdaptiveStep, RAdaptiveGrad
export RRTR, RRTRGrad
export initial_radius, update_radius!, reset_rule!
export needs_retrospective, is_criticality_anchored, retrospective_ratio
export asymptotic_regime, validate_thresholds, check_factors, check_bounds

# second order
export SecondOrder, RGradTau, RGradCappedTau, RDFOTau
export RAdaptiveGradTau, RRTRGradTau
export criticality, needs_curvature, tau_criticality
export lambda_min_estimate, curvature_estimate, EigenPoint, second_order_status

# ---- Axis 2: model Hessians -------------------------------------------------
export ModelHessian
export ExactHessian, LBFGSModel, SR1Model, ScaledIdentity, SPDTarget
export BHHHModel, BHHH2Model, GaussNewtonModel, OuterProductOperator
export hessian_op, dense_hessian, model_hprod!, update_model!, reset_model!
export phi_target, reports_negative_curvature, model_eltype

# ---- Axis 3: subproblem solvers ---------------------------------------------
export SubproblemSolver, SubWorkspace
export SteihaugCG, ExactMS, KrylovCG, KrylovCGLanczos
export solve_subproblem!, cg_step_info, returns_hprod, needs_eigenvector

# ---- Axis 4: sampling -------------------------------------------------------
export SamplingRule, SamplingState, SampleStats, batch_stats
export FullBatch, FixedSample, RadiusProportional, NormTest, GeometricSample
export InnerProductTest, OrthogonalityTest, AugmentedInnerProduct
export SequentialEstimation
export grad_sample_size, obj_sample_size, couples_to_radius, needs_scores
export sample_cap, requires_finite_population, reset_sampling_rule!

# sampled problems and oracles
export FiniteSum, PerturbedSum, PerturbedExpectation, GaussianDraw
export batch_obj, batch_grad!, batch_hess, grad_variance, obj_variance, draw_batch
export scores, loss_terms, score_matrix
export SampledNLP, ExpectationNLP, FiniteSumNLP, FullBatchNLP, LikelihoodNLP
export resample!, update_variances!, record_prediction!, samples_used
export reset_sampling!, population_cap, true_objective, true_gradient

# likelihood and least-squares problems
export residuals, jacobian, information_identity_error
export LogisticRegression, MLPClassifier, init_params, accuracy, β_true
export LeastSquares, linear_least_squares, exponential_fit, x_true, gauss_newton_error

# ---- The three solvers ------------------------------------------------------
export TRParams, TRResult, AbstractTRSolver
export DeterministicTRSolver, ExpectationTRSolver, FiniteSumTRSolver
export tr_solve, model_grad_evals

# ---- Benchmarking -----------------------------------------------------------
export performance_profile, data_profile, profile_to_pgfplots
export run_matrix, summarise, TRConfig, sweep_configs

# =============================================================================
# Registration with SolverCore
# =============================================================================

"""
    __init__()

Register `:second_order` with `SolverCore.STATUSES`.

`set_status!` validates against that dictionary and rejects anything not in it, so
without this the second-order run — the one requested with `tol_H > 0` — fails at the
moment it succeeds.

It has to happen here rather than at top level: `STATUSES` belongs to another module,
and a mutation performed during precompilation is not preserved in the cache image.
`get!` rather than assignment so that a future SolverCore defining its own
`:second_order` wins, and so that reloading is idempotent.
"""
function __init__()
    if isdefined(SolverCore, :STATUSES) && SolverCore.STATUSES isa AbstractDict
        get!(SolverCore.STATUSES, :second_order,
             "second-order critical point: ‖g‖ ≤ tol and λ_min(B) ≥ -tol_H")
    else
        @warn "SolverCore.STATUSES not found; a run with tol_H > 0 will fail when it " *
              "tries to report :second_order. Set tol_H = -1, or report the status " *
              "from :lambda_min_trajectory instead." maxlog = 1
    end
    return nothing
end

"""
    greet()

One-line description of the package.
"""
greet() = println("TrustRegionRadius: a testbed for trust-region radius update mechanisms.")

end # module TrustRegionRadius
