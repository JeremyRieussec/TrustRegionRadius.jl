using Test
using TrustRegionRadius
using LinearAlgebra
using ADNLPModels
using Aqua

@testset "TrustRegionRadius.jl" begin

    # ==========================================================================
    # §1.1 — Radius update rules (no CUTEst, no NLP model required)
    # ==========================================================================
    @testset "1.1 Radius update rules" begin

        η₁ = 0.1
        η₂ = 0.9

        # ------------------------------------------------------------------
        # R1 — Classical multiplicative update
        # ------------------------------------------------------------------
        @testset "R1 ClassicalUpdate" begin
            for (γ₁, γ₂, γ₃, label) in [
                (0.25, 0.50, 2.0, "standard"),
                (0.10, 0.40, 3.0, "aggressive contraction"),
                (0.30, 0.80, 4.0, "aggressive expansion"),
            ]
                r  = R1ClassicalUpdate(γ₁, γ₂, γ₃)
                Δ  = 1.0
                sn = 0.8   # step norm (unused by R1)
                go = 1.2   # g_norm_old (unused by R1)
                gn = 0.9   # g_norm_new (unused by R1)

                @testset "R1 $label — unsuccessful (ρ < η₁)" begin
                    Δ_new = update_radius!(r, Δ, -0.5, η₁, η₂, sn, go, gn)
                    @test Δ_new == γ₁ * Δ
                    @test Δ_new < Δ
                    @test Δ_new > 0.0
                end

                @testset "R1 $label — moderate (η₁ ≤ ρ < η₂)" begin
                    Δ_new = update_radius!(r, Δ, 0.5, η₁, η₂, sn, go, gn)
                    @test Δ_new == γ₂ * Δ
                    @test Δ_new > 0.0
                end

                @testset "R1 $label — very successful (ρ ≥ η₂)" begin
                    Δ_new = update_radius!(r, Δ, 0.95, η₁, η₂, sn, go, gn)
                    @test Δ_new == γ₃ * Δ
                    @test Δ_new >= Δ    # γ₃ ≥ 1
                    @test Δ_new > 0.0
                end

                @testset "R1 $label — lower bound Δ > 0" begin
                    @test update_radius!(r, Δ, -100.0, η₁, η₂, sn, go, gn) > 0.0
                    @test update_radius!(r, Δ,   0.0,  η₁, η₂, sn, go, gn) > 0.0
                    @test update_radius!(r, Δ,  10.0,  η₁, η₂, sn, go, gn) > 0.0
                end
            end
        end

        # ------------------------------------------------------------------
        # R2 — Step-size proportional update
        # ------------------------------------------------------------------
        @testset "R2 StepSizeUpdate" begin
            for (γ₁, γ₂, γ₃, label) in [
                (0.25, 0.80, 2.0, "standard"),
                (0.10, 0.60, 3.0, "aggressive contraction"),
                (0.30, 1.50, 5.0, "aggressive expansion"),
            ]
                r      = R2StepSizeUpdate(γ₁, γ₂, γ₃)
                Δ      = 1.0
                s_norm = 1.0   # use s_norm = Δ (boundary step)
                go     = 1.2
                gn     = 0.9

                @testset "R2 $label — unsuccessful (ρ < η₁) contracts Δ" begin
                    Δ_new = update_radius!(r, Δ, -0.5, η₁, η₂, s_norm, go, gn)
                    @test Δ_new == γ₁ * Δ
                    @test Δ_new < Δ
                    @test Δ_new > 0.0
                end

                @testset "R2 $label — moderate (η₁ ≤ ρ < η₂) proportional to ‖s‖" begin
                    Δ_new = update_radius!(r, Δ, 0.5, η₁, η₂, s_norm, go, gn)
                    @test Δ_new ≈ γ₂ * s_norm
                    @test Δ_new > 0.0
                end

                @testset "R2 $label — very successful (ρ ≥ η₂) larger factor than moderate" begin
                    Δ_mod = update_radius!(r, Δ, 0.5,  η₁, η₂, s_norm, go, gn)
                    Δ_vs  = update_radius!(r, Δ, 0.95, η₁, η₂, s_norm, go, gn)
                    @test Δ_vs ≈ γ₃ * s_norm
                    @test Δ_vs > Δ_mod   # γ₃ > γ₂
                    @test Δ_vs > 0.0
                end

                @testset "R2 $label — lower bound Δ > 0" begin
                    @test update_radius!(r, Δ, -100.0, η₁, η₂, s_norm, go, gn) > 0.0
                    @test update_radius!(r, Δ,   10.0, η₁, η₂, s_norm, go, gn) > 0.0
                end
            end
        end

        # ------------------------------------------------------------------
        # R3 — DFO-like gradient-radius comparison
        # ------------------------------------------------------------------
        @testset "R3 DFOLikeUpdate" begin
            for (γ₁, γ₂, γ₃, ζ, label) in [
                (0.25, 0.50, 2.0, 1.0, "standard (ζ=1)"),
                (0.10, 0.50, 2.0, 0.5, "small ζ=0.5"),
                (0.25, 0.90, 4.0, 2.0, "large ζ=2"),
            ]
                r  = R3DFOLikeUpdate(γ₁, γ₂, γ₃, ζ)
                Δ  = 1.0
                sn = 0.8
                gn = 0.9

                @testset "R3 $label — unsuccessful (ρ < η₁)" begin
                    Δ_new = update_radius!(r, Δ, -0.5, η₁, η₂, sn, 1.2, gn)
                    @test Δ_new == γ₁ * Δ
                    @test Δ_new < Δ
                    @test Δ_new > 0.0
                end

                @testset "R3 $label — successful, Δ ≤ ζ·‖g‖ → expand (γ₃)" begin
                    # Choose g_old so that Δ < ζ * g_old
                    g_old_large = 2.0 * Δ / ζ + 0.1
                    Δ_new = update_radius!(r, Δ, 0.5, η₁, η₂, sn, g_old_large, gn)
                    @test Δ_new == γ₃ * Δ
                    @test Δ_new >= Δ    # γ₃ ≥ 1
                    @test Δ_new > 0.0
                end

                @testset "R3 $label — successful, Δ > ζ·‖g‖ → no expand (γ₂)" begin
                    # Choose g_old so that Δ > ζ * g_old
                    g_old_small = 0.1 * Δ / ζ
                    Δ_new = update_radius!(r, Δ, 0.5, η₁, η₂, sn, g_old_small, gn)
                    @test Δ_new == γ₂ * Δ
                    @test Δ_new > 0.0
                end

                @testset "R3 $label — lower bound Δ > 0" begin
                    @test update_radius!(r, Δ, -100.0, η₁, η₂, sn, 1.0, gn) > 0.0
                    @test update_radius!(r, Δ,   10.0, η₁, η₂, sn, 1.0, gn) > 0.0
                end
            end
        end

        # ------------------------------------------------------------------
        # R4 — Relative-gradient update  (mutable μ: fresh rule per test)
        # ------------------------------------------------------------------
        @testset "R4 RelativeGradUpdate" begin
            for (γ₁, γ₂, μ₀, label) in [
                (0.25, 2.0, 1.0, "standard"),
                (0.10, 3.0, 0.5, "aggressive contraction"),
                (0.30, 4.0, 2.0, "large initial μ"),
            ]
                g_old = 1.2
                g_new = 0.9
                Δ = μ₀ * g_old   # Δ = μ * ‖g‖ invariant at start

                @testset "R4 $label — unsuccessful (ρ < η₁) contracts μ" begin
                    r = R4RelativeGradUpdate(γ₁, γ₂, μ₀)
                    Δ_new = update_radius!(r, Δ, -0.5, η₁, η₂, 0.3 * Δ, g_old, g_new)
                    @test r.μ ≈ γ₁ * μ₀
                    @test Δ_new ≈ r.μ * g_new
                    @test Δ_new > 0.0
                end

                @testset "R4 $label — moderate step, interior (ρ in [η₁,η₂], ‖s‖ ≤ 0.5Δ): μ unchanged" begin
                    r = R4RelativeGradUpdate(γ₁, γ₂, μ₀)
                    Δ_new = update_radius!(r, Δ, 0.5, η₁, η₂, 0.3 * Δ, g_old, g_new)
                    @test r.μ ≈ μ₀
                    @test Δ_new ≈ μ₀ * g_new
                    @test Δ_new > 0.0
                end

                @testset "R4 $label — very successful + boundary step (ρ ≥ η₂, ‖s‖ > 0.5Δ): μ expands" begin
                    r = R4RelativeGradUpdate(γ₁, γ₂, μ₀)
                    s_bnd = 0.9 * Δ   # > 0.5 * Δ
                    Δ_new = update_radius!(r, Δ, 0.95, η₁, η₂, s_bnd, g_old, g_new)
                    @test r.μ ≈ γ₂ * μ₀
                    @test Δ_new ≈ r.μ * g_new
                    @test Δ_new > 0.0
                end

                @testset "R4 $label — very successful + interior step (ρ ≥ η₂, ‖s‖ ≤ 0.5Δ): μ unchanged" begin
                    r = R4RelativeGradUpdate(γ₁, γ₂, μ₀)
                    s_int = 0.3 * Δ   # ≤ 0.5 * Δ
                    Δ_new = update_radius!(r, Δ, 0.95, η₁, η₂, s_int, g_old, g_new)
                    @test r.μ ≈ μ₀
                    @test Δ_new ≈ μ₀ * g_new
                    @test Δ_new > 0.0
                end

                @testset "R4 $label — invariant Δ = μ · ‖g‖ after every update" begin
                    r = R4RelativeGradUpdate(γ₁, γ₂, μ₀)
                    for (ρ_test, sn_test) in [(-0.5, 0.2Δ), (0.5, 0.3Δ), (0.95, 0.8Δ)]
                        Δ_new = update_radius!(r, Δ, ρ_test, η₁, η₂, sn_test, g_old, g_new)
                        @test Δ_new ≈ r.μ * g_new
                    end
                end

                @testset "R4 $label — lower bound Δ > 0" begin
                    r = R4RelativeGradUpdate(γ₁, γ₂, μ₀)
                    @test update_radius!(r, Δ, -100.0, η₁, η₂, 0.5, g_old, g_new) > 0.0
                end
            end
        end

    end  # 1.1 Radius update rules


    # ==========================================================================
    # §1.2 — Truncated CG subproblem solver
    # ==========================================================================
    @testset "1.2 Truncated CG subproblem solver" begin

        n     = 5
        A_pd  = Diagonal(Float64.(1:n))   # eigenvalues 1…5, ‖H‖ = 5
        b_pd  = ones(n)
        x0_pd = 2.0 * ones(n)
        g_pd  = A_pd * x0_pd + b_pd       # gradient at x0_pd
        H_norm_pd = Float64(n)            # spectral norm = max eigenvalue = 5

        nlp_pd = ADNLPModel(x -> 0.5 * dot(x, A_pd * x) + dot(b_pd, x), x0_pd,
                            name = "PDQuadratic")

        # Compute quadratic model decrease  m(0) - m(s) = -gᵀs - ½sᵀHs
        function qmodel_decrease(g, s, Hs)
            return -dot(g, s) - 0.5 * dot(s, Hs)
        end

        # Cauchy decrease lower bound
        function cauchy_lower_bound(g_norm, H_norm, Δ; κ = 0.1)
            return (κ / 2) * g_norm * min(g_norm / H_norm, Δ)
        end

        # ------------------------------------------------------------------
        # Test 1: Large Δ → step matches unconstrained Newton step
        # ------------------------------------------------------------------
        @testset "PD quadratic, large Δ: step ≈ unconstrained Newton direction" begin
            Δ_large = 1e4
            s, on_bnd, _ = truncated_cg_steihaug(nlp_pd, x0_pd, g_pd, Δ_large)
            s_newton = -(A_pd \ g_pd)     # exact Newton step

            @test norm(s - s_newton) < 1e-6
            @test norm(s) <= Δ_large + 1e-10
            Hs = A_pd * s
            md = qmodel_decrease(g_pd, s, Hs)
            @test md >= 0.0
            @test md >= cauchy_lower_bound(norm(g_pd), H_norm_pd, Δ_large)
        end

        # ------------------------------------------------------------------
        # Test 2: Small Δ → step on trust-region boundary
        # ------------------------------------------------------------------
        @testset "PD quadratic, small Δ: ‖s‖ ≤ Δ and model decreases" begin
            Δ_small = 0.1
            s, on_bnd, _ = truncated_cg_steihaug(nlp_pd, x0_pd, g_pd, Δ_small)

            @test norm(s) <= Δ_small + 1e-10
            Hs = A_pd * s
            md = qmodel_decrease(g_pd, s, Hs)
            @test md >= 0.0
            @test md >= cauchy_lower_bound(norm(g_pd), H_norm_pd, Δ_small)
        end

        # ------------------------------------------------------------------
        # Test 3: Indefinite quadratic → solver hits trust-region boundary
        # ------------------------------------------------------------------
        @testset "Indefinite quadratic: on_boundary == true" begin
            A_indef   = Diagonal([-5.0; fill(1.0, n - 1)])
            H_norm_in = 5.0
            x0_in     = ones(n)
            g_in      = A_indef * x0_in   # gradient at x0_in (no linear term)

            nlp_indef = ADNLPModel(x -> 0.5 * dot(x, A_indef * x), x0_in,
                                   name = "IndefiniteQuadratic")
            Δ = 2.0
            s, on_bnd, _ = truncated_cg_steihaug(nlp_indef, x0_in, g_in, Δ)

            @test on_bnd == true
            @test norm(s) <= Δ + 1e-10
            Hs = A_indef * s
            md = qmodel_decrease(g_in, s, Hs)
            @test md >= 0.0
            @test md >= cauchy_lower_bound(norm(g_in), H_norm_in, Δ)
        end

        # ------------------------------------------------------------------
        # Test 4: Zero gradient → s = 0
        # Note: a bug exists in the CG for g=0 (NaN from find_trust_region_boundary
        # when d=0). This test documents the expected behaviour; it will fail
        # until the zero-gradient guard is added to truncated_cg_steihaug.
        # ------------------------------------------------------------------
        @testset "Zero gradient: s = 0 expected (known zero-gradient bug)" begin
            g_zero = zeros(n)
            s, _, _ = truncated_cg_steihaug(nlp_pd, x0_pd, g_zero, 1.0)
            @test norm(s) < 1e-10
        end

    end  # 1.2 Truncated CG


    # ==========================================================================
    # §1.3 — Generic trust-region solver on analytical problems
    # ==========================================================================
    @testset "1.3 Generic solver on analytical problems" begin

        params = TRSolverParams(η₁ = 0.1, η₂ = 0.9, Δ₀ = 1.0,
                                max_iterations = 10_000, tol = 1e-5)

        # Prototype rules (deep-copied per (problem, rule) pair)
        rule_protos = [
            ("R1", R1ClassicalUpdate(0.25, 0.50, 2.0)),
            ("R2", R2StepSizeUpdate(0.25, 0.80, 2.0)),
            ("R3", R3DFOLikeUpdate(0.25, 0.50, 2.0, 1.0)),
            ("R4", R4RelativeGradUpdate(0.25, 2.0, 1.0)),
        ]

        # Problem factories (called fresh for each (problem, rule) pair)
        function make_rosenbrock()
            ADNLPModel(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
                       [-1.2; 1.0],
                       name = "Rosenbrock2")
        end

        function make_quadratic10()
            A = Diagonal(Float64.(1:10))
            b = ones(10)
            ADNLPModel(x -> 0.5 * dot(x, A * x) + dot(b, x),
                       ones(10),
                       name = "Quadratic10")
        end

        function make_ext_rosenbrock()
            x0 = [iseven(i) ? 1.0 : -1.2 for i in 1:20]
            ADNLPModel(
                x -> sum((1 - x[2i-1])^2 + 100 * (x[2i] - x[2i-1]^2)^2
                         for i in 1:10),
                x0,
                name = "ExtRosenbrock20",
            )
        end

        problems = [
            ("Rosenbrock (n=2)",          make_rosenbrock),
            ("Quadratic (n=10)",          make_quadratic10),
            ("Extended Rosenbrock (n=20)", make_ext_rosenbrock),
        ]

        for (prob_name, make_nlp) in problems
            @testset "$prob_name" begin
                for (rule_name, rule_proto) in rule_protos
                    rule = deepcopy(rule_proto)
                    nlp  = make_nlp()
                    @testset "$rule_name" begin
                        out = trust_region_solver(nlp, rule, params)

                        @test out.status == :solved
                        @test out.final_grad_norm <= 1e-5
                        @test length(out.delta_trajectory)     == out.iterations + 1
                        @test length(out.grad_norm_trajectory) == out.iterations + 1
                        @test length(out.obj_trajectory)       == out.iterations + 1
                        # Objective is non-increasing (rejected steps repeat the same value)
                        @test all(diff(out.obj_trajectory) .<= 0.0)
                    end
                end
            end
        end

    end  # 1.3 Generic solver


    # ==========================================================================
    # §1.4 — CUTEst integration tests  (guarded by ENV["CUTEST_AVAILABLE"])
    # ==========================================================================
    if haskey(ENV, "CUTEST_AVAILABLE")
        @testset "1.4 CUTEst integration" begin
            using CUTEst

            params_c = TRSolverParams(η₁ = 0.1, η₂ = 0.9, Δ₀ = 1.0,
                                      max_iterations = 10_000, tol = 1e-5)

            rules_c = [
                ("R1", R1ClassicalUpdate(0.25, 0.50, 2.0)),
                ("R2", R2StepSizeUpdate(0.25, 0.80, 2.0)),
                ("R3", R3DFOLikeUpdate(0.25, 0.50, 2.0, 1.0)),
                ("R4", R4RelativeGradUpdate(0.25, 2.0, 1.0)),
            ]

            valid_statuses = (:solved, :max_iter, :failure)

            for prob_name in ["ROSENBR", "BROWN4", "PENALTY1"]
                @testset "CUTEst $prob_name" begin
                    for (rule_name, rule_proto) in rules_c
                        @testset "$rule_name" begin
                            rule = deepcopy(rule_proto)
                            nlp  = CUTEstModel(prob_name)
                            try
                                out = trust_region_solver(nlp, rule, params_c)

                                @test out.status in valid_statuses
                                @test length(out.delta_trajectory)     == out.iterations + 1
                                @test length(out.grad_norm_trajectory) == out.iterations + 1
                                @test length(out.obj_trajectory)       == out.iterations + 1
                                if out.status == :solved
                                    @test out.final_grad_norm <= 1e-5
                                end
                            finally
                                finalize(nlp)
                            end
                        end
                    end
                end
            end

        end
    end  # 1.4 CUTEst


    # ==========================================================================
    # §1.5 — TROutput struct completeness
    # ==========================================================================
    @testset "1.5 TROutput struct completeness" begin

        params = TRSolverParams(η₁ = 0.1, η₂ = 0.9, Δ₀ = 1.0,
                                max_iterations = 10_000, tol = 1e-5)
        rule   = R1ClassicalUpdate(0.25, 0.5, 2.0)
        nlp    = ADNLPModel(x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
                            [-1.2; 1.0])
        out    = trust_region_solver(nlp, rule, params)

        @test out.iterations >= 0
        @test out.f_evals    >= out.iterations
        @test out.f_evals    >= 1
        @test out.final_grad_norm >= 0.0
        @test !isnan(out.final_grad_norm)
        @test out.final_delta > 0.0
        @test !isnan(out.final_delta)
        @test !isinf(out.final_delta)
        @test length(out.delta_trajectory)     == out.iterations + 1
        @test length(out.grad_norm_trajectory) == out.iterations + 1
        @test length(out.obj_trajectory)       == out.iterations + 1
        @test out.solve_time > 0.0
        @test all(!isnan,  out.delta_trajectory)
        @test all(!isinf,  out.delta_trajectory)
        @test all(>(0.0),  out.delta_trajectory)
        @test all(!isnan,  out.grad_norm_trajectory)
        @test all(!isnan,  out.obj_trajectory)
        @test all(diff(out.obj_trajectory) .<= 0.0)

    end  # 1.5 TROutput completeness


    # ==========================================================================
    # §2.1 — Aqua code quality
    # ==========================================================================
    @testset "2.1 Aqua code quality" begin
        Aqua.test_all(TrustRegionRadius)
    end


    # ==========================================================================
    # §2.2 — JET type stability  (optional, guarded by ENV["RUN_JET"])
    # ==========================================================================
    if haskey(ENV, "RUN_JET")
        @testset "2.2 JET type stability" begin
            using JET
            JET.test_package(TrustRegionRadius; target_defined_modules = true)
        end
    end

end  # TrustRegionRadius.jl
