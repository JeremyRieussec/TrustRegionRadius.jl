# The sampling layer: problems, sampling rules, and the SampledNLP oracle.
#
# The property that matters most is the one checked last: with common random
# numbers the noise in f̂(x) − f̂(x+s) shrinks with the step, and without them it
# does not. Every mechanism stalls at small radii in the second case, for reasons
# that have nothing to do with the mechanism.

@testset "stochastic" begin
    quad(n = 4) = ADNLPModel(x -> 0.5 * sum(collect(1.0:n) .* (x .- 1.0).^2),
                             zeros(n), name = "quad")

    @testset "PerturbedSum has the base model as its exact mean" begin
        base = quad(); p = PerturbedSum(base, 200; σg = 1.0, seed = 3)
        x = [0.3, -0.7, 1.4, 0.2]
        full = collect(1:200)
        @test batch_obj(p, x, full) ≈ obj(base, x)          rtol = 1e-10
        g = zeros(4); batch_grad!(p, x, full, g)
        @test g ≈ grad(base, x)                              rtol = 1e-10
        @test true_objective(p, x) ≈ obj(base, x)
        @test true_gradient(p, x)  ≈ grad(base, x)
        # A subsample is biased away from it, by an amount that shrinks with N.
        errs = Float64[]
        for N in (4, 64, 1024)
            rng = MersenneTwister(1)
            e = mean(1:40) do _
                b = rand(rng, 1:200, N); gg = zeros(4)
                batch_grad!(p, x, b, gg); norm(gg - grad(base, x))
            end
            push!(errs, e)
        end
        @test errs[1] > errs[2] > errs[3]                    # Monte Carlo rate
        @test errs[3] < errs[1] / 4
    end

    @testset "sampling rules" begin
        st = SamplingState(3, 0.1, 2.0, 4.0, 9.0)            # σg = 2, σf = 3

        @test grad_sample_size(FixedSample(32), st) == 32
        @test obj_sample_size(FixedSample(32; N_obj = 8), st) == 8
        @test !couples_to_radius(FixedSample(32))

        # N ≥ (σ/(κΔ))² = (2/0.1)² = 400, and (3/0.01)² = 90 000
        rp = RadiusProportional(κ_g = 1.0, κ_f = 1.0, N_max = 10^7)
        @test grad_sample_size(rp, st) == 400
        @test obj_sample_size(rp, st)  == 90_000
        @test couples_to_radius(rp)

        # N ~ Δ^{-2}: halving the radius quadruples the sample size.
        st2 = SamplingState(3, 0.05, 2.0, 4.0, 9.0)
        @test grad_sample_size(rp, st2) == 4 * grad_sample_size(rp, st)

        # Norm test: σ²/(θ²‖g‖²) = 4/(0.25·4) = 4, and it never reads Δ.
        nt = NormTest(θ = 0.5)
        @test grad_sample_size(nt, st) == 4
        @test grad_sample_size(nt, SamplingState(3, 1e-8, 2.0, 4.0, 9.0)) ==
              grad_sample_size(nt, st)
        @test !couples_to_radius(nt)

        @test grad_sample_size(GeometricSample(N₀ = 8, rate = 2.0), st) == 64
        @test_throws ArgumentError RadiusProportional(κ_g = 0.0)
        @test_throws ArgumentError NormTest(θ = -1.0)
        @test_throws ArgumentError GeometricSample(rate = 0.5)
    end

    @testset "SampledNLP satisfies the NLP interface" begin
        base = quad(); p = PerturbedSum(base, 500; σg = 1.0, seed = 5)
        m = SampledNLP(p, FixedSample(500))
        x = [0.3, -0.7, 1.4, 0.2]
        resample!(m, 1, 1.0, 1.0)
        @test length(m.batch_g) == 500
        @test obj(m, x) isa Real
        g = zeros(4); grad!(m, x, g); @test length(g) == 4
        @test size(Matrix(hess(m, x))) == (4, 4)
        Hv = zeros(4); hprod!(m, x, ones(4), Hv); @test length(Hv) == 4
        @test samples_used(m).total == 1000                   # 500 grad + 500 obj
    end

    @testset "the solver counts samples, not just iterations" begin
        base = quad(); p = PerturbedSum(base, 2_000; σg = 0.5, seed = 7)
        m = SampledNLP(p, FixedSample(32))
        st = tr_solve(m; rule = RDelta(), model = ExactHessian(),
                      subsolver = ExactMS(), trace = true,
                      params = TRParams(tol = 1e-4, max_iterations = 60))
        ss = st.solver_specific
        @test haskey(ss, :grad_sample_trajectory)
        @test haskey(ss, :samples_total)
        @test length(ss[:grad_sample_trajectory]) == st.iter
        @test all(==(32), ss[:grad_sample_trajectory])
        @test ss[:samples_total] == 64 * st.iter              # grad + obj

        # A deterministic run is untouched: no sampling keys, no resampling.
        st2 = tr_solve(quad(); rule = RDelta(), trace = true,
                       params = TRParams(tol = 1e-8))
        @test !haskey(st2.solver_specific, :samples_total)
    end

    @testset "radius-proportional sampling grows N as Δ falls" begin
        base = quad(); p = PerturbedSum(base, 20_000; σg = 1.0, seed = 11)
        m = SampledNLP(p, RadiusProportional(κ_g = 1.0, κ_f = 1.0, N_max = 20_000))
        st = tr_solve(m; rule = RDFO(ζ = 1.0), model = ExactHessian(),
                      subsolver = ExactMS(), trace = true,
                      params = TRParams(tol = 1e-6, max_iterations = 40))
        ss = st.solver_specific
        N = ss[:grad_sample_trajectory]; Δ = ss[:delta_trajectory]
        @test length(N) >= 5
        # N_k and Δ_k move in opposite directions: the mechanism pays for its own
        # radius policy. Checked on the run's own trajectory rather than asserted.
        @test cor(log.(max.(N, 1)), log.(max.(Δ[1:length(N)], 1e-300))) < -0.5
    end

    @testset "common random numbers keep ρ̂ usable at small radii" begin
        # With a shared batch, f̂(x) − f̂(x+s) is an average of F_i(x) − F_i(x+s)
        # and shrinks with ‖s‖. With independent batches it does not, so ρ̂ is
        # noise once the step is smaller than the sampling error — and every
        # mechanism then stalls for a reason external to it.
        base = quad(); p = PerturbedSum(base, 5_000; σg = 1.0, seed = 13)
        rng = MersenneTwister(2)
        x = [0.5, 0.5, 0.5, 0.5]; d = [1.0, 0.0, 0.0, 0.0]

        for α in (1e-1, 1e-3)
            s = α .* d
            crn = Float64[]; ind = Float64[]
            for _ in 1:200
                b1 = rand(rng, 1:5_000, 64); b2 = rand(rng, 1:5_000, 64)
                truth = true_objective(p, x) - true_objective(p, x .+ s)
                push!(crn, batch_obj(p, x, b1) - batch_obj(p, x .+ s, b1) - truth)
                push!(ind, batch_obj(p, x, b1) - batch_obj(p, x .+ s, b2) - truth)
            end
            # Shared batch: error scales with ‖s‖. Independent: it does not.
            @test std(crn) < std(ind)
            α < 1e-2 && @test std(crn) < 0.1 * std(ind)
        end
    end
end
