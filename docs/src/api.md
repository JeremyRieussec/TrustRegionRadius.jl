# API reference

The full docstring index, grouped by area. Each group links to the page where those
docstrings are written and discussed; this page is the lookup table, not the tutorial.

`TRParams` is documented under [Thresholds and factors](thresholds.md), the three
solvers under [Problem classes](problem_classes.md), and the trace alignment convention
under [Diagnostics](diagnostics.md).

## Entry point

```@docs
TRResult
AbstractTRSolver
tr_solve
model_grad_evals
```

## Extension points

These are exported so that a rule, a model Hessian or a subproblem solver defined
outside the package can hook into it. They are validators and traits rather than
user-facing API: you call them when *writing* an extension, not when solving a problem.

- Rule authors: [`check_factors`](@ref), [`check_bounds`](@ref),
  [`validate_thresholds`](@ref), [`initial_radius`](@ref), [`update_radius!`](@ref),
  [`reset_rule!`](@ref), [`asymptotic_regime`](@ref), [`is_criticality_anchored`](@ref),
  [`needs_retrospective`](@ref), [`last_branch`](@ref).
- Model-Hessian authors: [`hessian_op`](@ref), [`dense_hessian`](@ref),
  [`model_hprod!`](@ref), [`update_model!`](@ref), [`reset_model!`](@ref),
  [`reports_negative_curvature`](@ref), [`model_eltype`](@ref),
  [`required_problem`](@ref).
- Subproblem-solver authors: [`solve_subproblem!`](@ref), [`SubWorkspace`](@ref),
  [`returns_hprod`](@ref), [`needs_eigenvector`](@ref).
- Problem and oracle authors: [`problem_class`](@ref), [`population`](@ref),
  [`n_terms`](@ref), [`has_scores`](@ref), [`has_truth`](@ref), [`full_batch`](@ref),
  [`underlying_problem`](@ref), [`check_model_problem`](@ref),
  [`check_rule_problem`](@ref), [`check_population_cap`](@ref), [`user_cap`](@ref).
- Sampling-rule authors: [`grad_sample_size`](@ref), [`obj_sample_size`](@ref),
  [`couples_to_radius`](@ref), [`needs_scores`](@ref), [`needs_paired`](@ref),
  [`requires_finite_population`](@ref), [`sample_cap`](@ref),
  [`reset_sampling_rule!`](@ref), [`supports_paired`](@ref).

## Index by area

### Problem classes and oracles

```@index
Pages = ["problem_classes.md"]
Order = [:type, :function]
```

### Radius mechanisms

```@index
Pages = ["rules.md"]
Order = [:type, :function]
```

### Thresholds and factors

```@index
Pages = ["thresholds.md"]
Order = [:type, :function]
```

### Model Hessians

```@index
Pages = ["models.md"]
Order = [:type, :function]
```

### Subproblem solvers

```@index
Pages = ["subsolvers.md"]
Order = [:type, :function]
```

### Sampling rules and stochastic oracles

```@index
Pages = ["stochastic.md"]
Order = [:type, :function]
```

### Likelihood and least squares

```@index
Pages = ["likelihood.md"]
Order = [:type, :function]
```

### Second-order variants

```@index
Pages = ["second_order.md"]
Order = [:type, :function]
```

### Diagnostics

```@index
Pages = ["diagnostics.md"]
Order = [:type, :function]
```

### Benchmarking

```@index
Pages = ["benchmarking.md"]
Order = [:type, :function]
```

### Entry point

```@index
Pages = ["api.md"]
Order = [:type, :function]
```
