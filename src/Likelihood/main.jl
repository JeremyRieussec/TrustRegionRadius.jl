# =============================================================================
# ───── src/Likelihood/main.jl

# Outer-product model Hessians and the likelihood problems they are defined for.
# After Sampling/: likelihood.jl dispatches score_matrix on FullBatchNLP and
# SampledNLP, and models.jl subtypes LikelihoodProblem.
#
#   likelihood.jl     BHHH, BHHH-2, Gauss-Newton, OuterProductOperator
#   least_squares.jl  LeastSquares <: NLSProblem
#   models.jl         LogisticRegression, MultinomialLogit, MLPClassifier
#   probit.jl         Probit; after models.jl, which declares β_true
#   linear_regression.jl  LinearRegression; after models.jl, for the same reason
# =============================================================================

include("likelihood.jl")
include("least_squares.jl")
include("models.jl")
include("probit.jl")
include("linear_regression.jl")