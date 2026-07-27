# TrustRegionRadius.jl

Companion code for *A survey of trust-region radius update mechanisms*.

See the [README](https://github.com/…/TrustRegionRadius.jl) for a tour, and
`INTEGRATION.md` for the design notes and the record of what changed when the
package was restructured.

## API

```@docs
RadiusRule
RDelta
RStep
RDFO
RGrad
RGradCapped
RAdaptiveStep
RAdaptiveGrad
RAdaptiveFanYuan
RRTR
RRTRGrad
initial_radius
update_radius!
reset_rule!
needs_retrospective
is_criticality_anchored
retrospective_ratio
```

```@docs
ModelHessian
ExactHessian
LBFGSModel
SR1Model
ScaledIdentity
SPDTarget
hessian_op
dense_hessian
model_hprod!
update_model!
reset_model!
phi_target
```

```@docs
SubproblemSolver
SteihaugCG
ExactMS
KrylovCG
KrylovCGLanczos
solve_subproblem!
cg_step_info
```

```@docs
TRParams
TRResult
TRSolver
tr_solve
```

```@docs
performance_profile
data_profile
profile_to_pgfplots
run_matrix
summarise
TRConfig
sweep_configs
```
