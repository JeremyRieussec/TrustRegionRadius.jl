```
TrustRegionRadius.jl/
│
├── Project.toml                        ← never uploaded; must list the new files? no —
│                                          Julia globs src/, but check [deps] and [compat]
├── README.md                           ← never uploaded
├── CITATION.cff                        ← never uploaded
├── MIGRATION.md                        ← never uploaded; index.md links to it.
│                                          The table in REFACTOR.md is what belongs here
│
├── src/
│   ├── TrustRegionRadius.jl      111   module: includes + 45 export lines
│   │
│   ├── Problems/                       ★ NEW — the class hierarchy everything checks against
│   │   ├── main.jl                 1
│   │   └── classes.jl            353   AbstractProblem … NLSProblem; traits;
│   │                                   required_problem / check_model_problem /
│   │                                   check_population_cap
│   │
│   ├── Radius_updates/                 axis 1 — the subject of the study
│   │   ├── main.jl                 2
│   │   ├── rules.jl              927   RadiusRule + 10 mechanisms, check_factors, check_bounds
│   │   └── retrospective.jl       65   retrospective_ratio
│   │
│   ├── Model_Hessians/                 axis 2
│   │   ├── main.jl                 1
│   │   └── model_hessian.jl      390   ExactHessian, LBFGS, SR1, ScaledIdentity, SPDTarget
│   │
│   ├── Subproblem/                     axis 3
│   │   ├── main.jl                 1
│   │   └── subproblem.jl         443   SteihaugCG, ExactMS, KrylovCG, KrylovCGLanczos,
│   │                                   SubWorkspace, cg_step_info
│   │
│   ├── Second_order/
│   │   ├── main.jl                 1
│   │   └── second_order.jl       486   SecondOrder wrapper, τ-rules, curvature_estimate,
│   │                                   EigenPoint
│   │
│   ├── Sampling/                       ★ was src/Stochastic/ — axis 4
│   │   ├── main.jl                 3
│   │   ├── problems.jl           452   FiniteSum, PerturbedSum, PerturbedExpectation,
│   │   │                               GaussianDraw, draw_batch, full_batch, scores
│   │   ├── rules.jl              635   SamplingRule + FullBatch + the 8; sample_cap/user_cap
│   │   └── oracles.jl            429   FullBatchNLP, SampledNLP,
│   │                                   ExpectationNLP, FiniteSumNLP
│   │
│   ├── Likelihood/
│   │   ├── main.jl                 3
│   │   ├── likelihood.jl         321   BHHH, BHHH2, GaussNewton, OuterProductOperator,
│   │   │                               information_identity_error
│   │   ├── least_squares.jl      249   LeastSquares <: NLSProblem, gauss_newton_error
│   │   └── models.jl             366   LogisticRegression, MLPClassifier <: LikelihoodProblem
│   │
│   ├── Trust-region/                   ★ was one solver.jl
│   │   ├── main.jl                 5
│   │   ├── common.jl             461   TRParams, TRCore, TRState, TRTrace, _tr_step!
│   │   ├── deterministic.jl      103   DeterministicTRSolver
│   │   ├── expectation.jl        141   ExpectationTRSolver
│   │   ├── finitesum.jl          107   FiniteSumTRSolver
│   │   └── entry.jl               83   tr_solve — dispatches on the oracle
│   │
│   └── Benchmark/
│       └── main.jl                     ← never uploaded. run_matrix, summarise, TRConfig,
│                                          sweep_configs, performance_profile, data_profile,
│                                          profile_to_pgfplots. Needs the migration table
│                                          if it names TRSolver or SampledNLP
│
├── test/
│   ├── Project.toml               14
│   ├── runtests.jl                25
│   ├── test_problem_classes.jl   222   ★ NEW — 97 assertions, runs first
│   ├── test_rules.jl             267
│   ├── test_models.jl            127
│   ├── test_subproblem.jl        168
│   ├── test_solver.jl            234
│   ├── test_profiles.jl           67   (Benchmark layer — untouched, unverified)
│   ├── test_thresholds.jl        166
│   ├── test_second_order.jl      288
│   ├── test_stochastic.jl        300
│   ├── test_likelihood.jl        322
│   ├── test_sampling.jl          302
│   └── test_exports.jl            56
│
├── docs/
│   ├── make.jl                    41   "The four axes"; new Problem classes page
│   └── src/
│       ├── index.md              106
│       ├── quickstart.md         227
│       ├── thresholds.md         151
│       ├── problem_classes.md    159   ★ NEW
│       ├── rules.md              190
│       ├── models.md             111
│       ├── subsolvers.md         100
│       ├── likelihood.md         148
│       ├── stochastic.md         222   retitled "Sampling rules"
│       ├── second_order.md       146
│       ├── api.md                 18
│       └── benchmarking.md             ← listed in make.jl, never uploaded.
│                                          The build fails on the missing file
│
├── notebooks/
│   └── tutorial.ipynb           1117   46 cells, 114 assertions
│
├── benchmark/                          ← never uploaded (experiment scripts;
│                                          quickstart.md references "Experiment 8")
│
└── (working notes, not part of the package)
    ├── audit.md                  455   pass 1 — findings
    ├── CHANGES.md                106   pass 2 — source rewrite
    ├── TEST_CHANGES.md           242   pass 3 — test suite
    └── REFACTOR.md               181   pass 4 — this split
```