# Experiment Summary

**Archived:** 2026-04-16 00:25:56

## Solver Parameters

| Parameter | Value |
|-----------|-------|
| `Delta0` | 1.0 |
| `eta1` | 0.1 |
| `eta2` | 0.9 |
| `max_iterations` | 10000 |
| `tol` | 1.0e-5 |

## Problem Selection (CUTEst)

| Criterion | Value |
|-----------|-------|
| `max_con` | 0 |
| `max_var` | 500 |
| `min_var` | 500 |

## Radius Update Rules

- **R1** (`R1ClassicalUpdate`): gamma1 = 0.25, gamma2 = 0.5, gamma3 = 2.0
- **R2** (`R2StepSizeUpdate`): gamma1 = 0.25, gamma2 = 0.8, gamma3 = 2.0

## Outputs

| Item | Count |
|------|-------|
| JLD2 result files | 8 |
| Figures           | 3 |
| Tables            | 1 |

### Figures

- `figures/exp1_data_profile.pdf`
- `figures/exp1_perf_profile_fevals.pdf`
- `figures/exp1_perf_profile_iter.pdf`

### Tables

- `tables/exp1_success_rate.txt`
