# =============================================================================
# ───── src/Sampling/main.jl

# Axis 4: sampling. Order matters here.
#
#   problems.jl  the sampled problems and the batch-evaluation interface
#   rules.jl     SamplingRule and the nine rules; needs SampledProblem
#   oracles.jl   the NLP oracles; constructors call check_rule_problem,
#                check_population_cap and requires_finite_population, so both
#                of the above must be loaded first
# =============================================================================

include("problems.jl")
include("rules.jl")
include("oracles.jl")