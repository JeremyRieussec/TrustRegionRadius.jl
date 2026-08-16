# Whether the trace was *recorded* correctly, as distinct from whether the
# diagnostics compute the right thing from a trace (test_diagnostics.jl).
#
# The one place the sampled and exact code paths can be compared iterate for
# iterate is FiniteSumNLP(prob, FullBatch()) against FullBatchNLP(prob): at
# N_k = M the finite-sum iteration IS the deterministic one, so every recorded
# quantity has a reference with no sampling in it. Anything the resampling and
# re-evaluation logic gets wrong shows up as a divergence in one trajectory.
#
# Comparing only status, iter and solution is what let an earlier bootstrap bug
# hide: a batch that is not quite the population still lands near the same
# solution by a different path.

@testset "recording" begin

    base = ADNLPModel(z -> 0.5 * sum(collect(1.0:4) .* (z .- 1.0).^2), zeros(4))
    fsum = PerturbedSum(base, 500; σg = 1.0, seed = 1)

    # Trajectories the patch added, over and above the six original ones.
    approx_keys = (:gamma_trajectory, :xi_trajectory, :cos_cauchy_trajectory,
                   :rho_tilde_trajectory)
    exact_keys  = (:branch_trajectory, :cg_iters_trajectory)

    # Element-wise, and NaN matches NaN.
    #
    # Two reasons not to use `≈` on the vectors directly. Several of the added
    # trajectories are NaN on the first iterations, where the quantity is not yet
    # defined (γ and ξ both compare against a previous iteration), and array `≈`
    # is false as soon as either side contains a NaN even when the two are
    # identical. And array `≈` is norm-based, so on a long trajectory one badly
    # wrong entry can hide inside the norm. Comparing entry by entry is the
    # stricter test as well as the correct one.
    same(a, b; rtol = 1e-8) =
        length(a) == length(b) &&
        all(isapprox(x, y; rtol = rtol, nans = true) for (x, y) in zip(a, b))

    @testset "FullBatch reproduces the deterministic solver exactly" begin
        p = TRParams(tol = 1e-8, max_iterations = 200)
        det = tr_solve(FullBatchNLP(fsum); rule = RDelta(), subsolver = SteihaugCG(),
                       params = p, trace = true)
        fs  = tr_solve(FiniteSumNLP(fsum, FullBatch()); rule = RDelta(),
                       subsolver = SteihaugCG(), params = p, trace = true)

        @test det.status === fs.status
        @test det.iter == fs.iter
        @test det.solution ≈ fs.solution rtol = 1e-8

        # The premise of the whole comparison, checked rather than assumed: if
        # any iteration drew less than the population these are two different
        # algorithms and agreement below would be luck.
        @test all(fs.solver_specific[:full_batch_trajectory])
        @test length(fs.solver_specific[:full_batch_trajectory]) == fs.iter

        # the six original trajectories
        @test det.solver_specific[:active_trajectory] ==
              fs.solver_specific[:active_trajectory]
        @test det.solver_specific[:accepted_trajectory] ==
              fs.solver_specific[:accepted_trajectory]
        for k in (:delta_trajectory, :obj_trajectory, :ratio_trajectory,
                  :step_trajectory)
            @test same(det.solver_specific[k], fs.solver_specific[k])
        end

        # and the ones the patch added, which nothing compared until now
        for k in approx_keys
            @test haskey(det.solver_specific, k)
            @test haskey(fs.solver_specific, k)
            @test same(det.solver_specific[k], fs.solver_specific[k])
        end
        for k in exact_keys
            @test det.solver_specific[k] == fs.solver_specific[k]
        end
    end

    @testset "a run with rejections exercises both branches" begin
        # Rosenbrock from the standard start rejects steps; the equivalence run
        # above is a convex quadratic that accepts nearly everything, so on its
        # own it never visits the rejected branch of the recording code.
        rosen = ADNLPModel(z -> (1 - z[1])^2 + 100 * (z[2] - z[1]^2)^2, [-1.2, 1.0])
        st = tr_solve(rosen; rule = RDelta(), subsolver = SteihaugCG(), trace = true,
                      params = TRParams(tol = 1e-8, max_iterations = 500))
        ss = st.solver_specific

        acc = ss[:accepted_trajectory]
        @test any(acc)                       # both branches, or the run proves nothing
        @test any(!, acc)

        nstep = length(ss[:step_trajectory])
        @test length(ss[:branch_trajectory]) == nstep
        @test sum(values(branch_counts(st))) == nstep

        # Every per-iteration trajectory has one entry per iteration, and every
        # state trajectory one more. An off-by-one here shifts every plot.
        for k in (:ratio_trajectory, :active_trajectory, :accepted_trajectory,
                  approx_keys..., exact_keys...)
            @test length(ss[k]) == nstep
        end
        for k in (:delta_trajectory, :grad_trajectory, :obj_trajectory)
            @test length(ss[k]) == nstep + 1
        end

        # A rejected iteration must contract, and must not move the iterate.
        Δ = ss[:delta_trajectory]
        for k in 1:nstep
            acc[k] || @test Δ[k + 1] < Δ[k]
        end

        # branch_counts only ever reports branches the rules can emit
        @test all(k -> k in (:contract, :shrink, :expand, :expand_capped, :hold),
                  keys(branch_counts(st)))
    end

    @testset "the equivalence holds on a rejecting problem too" begin
        # The convex quadratic accepts almost every step, so the comparison above
        # never checks that the *rejected* path records identically. Force
        # rejections into the equivalence by starting far out with a tight radius.
        hard = ADNLPModel(z -> (1 - z[1])^2 + 100 * (z[2] - z[1]^2)^2, [-1.2, 1.0])
        hsum = PerturbedSum(hard, 200; σg = 0.0, seed = 4)   # σg = 0: mean is exact
        p = TRParams(tol = 1e-6, max_iterations = 300, Δ0 = 2.0)
        det = tr_solve(FullBatchNLP(hsum); rule = RDelta(), subsolver = SteihaugCG(),
                       params = p, trace = true)
        fs  = tr_solve(FiniteSumNLP(hsum, FullBatch()); rule = RDelta(),
                       subsolver = SteihaugCG(), params = p, trace = true)

        @test any(!, det.solver_specific[:accepted_trajectory])   # it does reject
        @test det.iter == fs.iter
        @test det.solver_specific[:accepted_trajectory] ==
              fs.solver_specific[:accepted_trajectory]
        @test det.solver_specific[:branch_trajectory] ==
              fs.solver_specific[:branch_trajectory]
        for k in (:delta_trajectory, :ratio_trajectory, :step_trajectory,
                  approx_keys...)
            @test same(det.solver_specific[k], fs.solver_specific[k])
        end
    end
end
