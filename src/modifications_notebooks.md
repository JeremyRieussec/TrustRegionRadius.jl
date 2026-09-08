``What to change, in order``
1. Raise PROBLEM_LIMIT and MAX_VAR, and re-run. Nothing below is worth doing on three two-variable problems.
2. Drop the solved filter from exp5 and exp7. In both, the excluded runs are the evidence. Report the status beside the statistic instead.
3. Restore the uncapped arm in exp4, or remove it from save_config. It should be restored: it is the only test of the one unconditional result in the survey.
4. exp7: report inactivity_index as the x-axis, keeping findfirst as a separate column if the difference is interesting; require a minimum tail length and report how many runs were dropped for being too short rather than printing --.
5. Report Σ Δ_k²/M_k in exp5 and exp2, per problem, and stop averaging series across problems.
6. Promote exp9's paired table into Part III. It already answers a Part II claim the draft leaves open.
7. exp12: pass the seeds to save_config, and add CertifiedDecrease to its rule list, which is where the §9.9 numbers should come from.


Fifteen files written. Six of the seven are done; item 6 is a paper change and belongs in T6/T7, so I have left the draft alone.

``Item 5`` turned out to need more than a table change. Σ Δ_k²/M_k needs M_k = L + max_{i≤k}‖H_i‖, and the package recorded no Hessian norm at all — so there was nothing to divide by. I added model_hessian_norm (dense opnorm for n ≤ 200, power iteration on H² otherwise), an opt-in hessian_norm = true flag threaded like true_curvature, and radius_sums in the diagnostics layer returning all three series with n beside them. It reports with L = 0, stated in the table footer, because the Lipschitz constant is not known and omitting it changes the constant rather than the convergence. exp2 and exp5 now print the three columns per run with the status beside them, and the footer says what a partial sum over forty iterations does and does not establish.

``Item 3`` was blocked by a type error, which is why it was commented out. The configs comprehension has one concrete closure type, so pushing a differently-shaped closure fails to convert — the abandoned println(typeof(configs)) debug lines beside it are the fingerprint. Declaring Tuple{String, Function}[...] fixes it. I started the uncapped arm at μ₀ = 0.1, far below any plausible κ̄, so the climb is what gets measured, and added an exp4_climb.txt table counting :expand against :expand_capped with k* — which is the log_{γ₃} bound of Corollary [mu three regimes] made checkable.

``Items 2 and 4`` removed both survivorship filters. exp5 now reports runs / solved / measured / mean tail / never inactive, so a trapped configuration appears as a never inactive count rather than vanishing, and a tail fraction is only computed on runs of at least 30 iterations — below that the last 10% is one or two iterations and the statistic is quantised to {0, ½, 1}. exp7 reports k* from inactivity_index as the rate axis and keeps 1st inactive as a separate column, since the two differ whenever the region releases and binds again; runs with fewer than six inactive iterations are counted as short instead of printed as --, and the per-rule table shows fitted / short / never so a median cannot quietly be taken over the mechanisms that go inactive early.

``Item 7`` adds CertifiedDecrease(p = 0.9) to the sampler list, and passes seed = collect(SEEDS) plus the rule factories to save_config — it was passing rule names as strings, so the archive recorded that they were strings and nothing else. All four exp12 examples are ScoredProblem subtypes, so pairing is supported on every cell.

``Two things to know before running.`` run_experiment now takes solver_kwargs, which is how exp2 and exp5 ask for the Hessian norm — but a cached record is returned as archived, so those two must be run with resume = false or into a fresh archive, otherwise the new column comes back empty for every problem already in data/. I documented that on run_experiment itself. And MAX_VAR = 100 with PROBLEM_LIMIT = 50 is a real campaign: at max_time = 120 per run and eight rules, budget hours, not minutes.

``T5 next:`` the experiments worth adding, now that the suite can measure k*, the branch counts, γ_k and the paired statistic.