# Experiment Summary

**Archive:** `exp_2026-04-16_02-08-34`
**Generated:** 2026-04-16 02:12:10

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
- **R3** (`R3DFOLikeUpdate`): gamma1 = 0.25, gamma2 = 0.5, gamma3 = 2.0, zeta = 1.0
- **R4** (`R4RelativeGradUpdate`): gamma1 = 0.25, gamma2 = 2.0, mu = 1.0

## Outputs

| Item | Count |
|------|-------|
| JLD2 result files | 16 |
| Figures           | 17 |
| Tables            | 4 |

### Figures

- `figures/exp1_data_profile.pdf`
- `figures/exp1_perf_profile_fevals.pdf`
- `figures/exp1_perf_profile_iter.pdf`
- `figures/exp2_cumsum_summary.pdf`
- `figures/exp2_delta_traj_GENROSE.pdf`
- `figures/exp2_delta_traj_GENROSEB.pdf`
- `figures/exp2_grad_traj_GENROSE.pdf`
- `figures/exp2_grad_traj_GENROSEB.pdf`
- `figures/exp2_ratio_traj_GENROSE.pdf`
- `figures/exp2_ratio_traj_GENROSEB.pdf`
- `figures/exp3_perf_profile_zeta.pdf`
- `figures/exp3_solverate_vs_zeta.pdf`
- `figures/exp4_mu_drift.pdf`
- `figures/exp4_perf_profile_mu.pdf`
- `figures/exp4_solverate_vs_mu.pdf`
- `figures/exp5_iter_comparison_illcond.pdf`
- `figures/exp5_perf_profile_illcond.pdf`

### Tables

- `tables/exp1_success_rate.txt`
- `tables/exp3_zeta_summary.txt`
- `tables/exp4_mu_summary.txt`
- `tables/exp5_illcond_summary.txt`
