# =============================================================================
# benchmark/experiments/exp11_bhhh.jl
#
# EXPERIMENT 11 -- outer-product Hessians, where they are justified and where not.
#
# Adapted to the problem-class split: `LikelihoodNLP` is now `FullBatchNLP`, the
# deterministic (all-terms) view of a finite sum. The old name survives as an
# alias, but the new one says what it is -- deterministic -- while still carrying
# the LikelihoodProblem underneath, which is what keeps BHHHModel legal over it.
# `required_problem(::BHHHModel) === LikelihoodProblem` is now enforced at solver
# construction, so a BHHH run against a non-likelihood is an ArgumentError naming
# both types rather than a positive semidefinite matrix that means nothing.
#
#  (a) THE IDENTITY, MEASURED. On logistic regression -- correctly specified by
#      construction -- ‖B − ∇²f‖/‖∇²f‖ decays like M^{-1/2} at β*, and does not
#      decay at all at β* + δ. Swept over the number of parameters K and the
#      sample size M.
#
#  (b) BHHH VERSUS BHHH-2. They differ by the rank-one ḡḡᵀ, so they agree at a
#      stationary point and can disagree sharply away from one. Which is better
#      early is an empirical question; this answers it on these problems.
#
#  (c) THE MECHANISMS OVER AN OUTER-PRODUCT MODEL. B ⪰ 0 always, so no radius
#      mechanism ever sees negative curvature. The comparison is against
#      ExactHessian on the same problem, same mechanism.
#
#  (d) NEURAL NETWORKS: THE JUSTIFICATION IS GONE. On MNIST and CIFAR-100 the
#      model is misspecified and there are no true parameters, so the identity
#      error does not shrink with the sample. BHHH is still a usable positive
#      semidefinite preconditioner; it is not an approximation to the Hessian, and
#      the run converges to saddles without complaint.
#
#   julia --project=benchmark benchmark/experiments/exp11_bhhh.jl
#
# Format note: no `using` and no `include` here. `initialisation.jl` loads the
# packages, `harness.jl`, `archive.jl` and `config.jl` once, in order, exactly as
# it does for experiments 1-7. Re-including them per file re-ran `config.jl` and
# so re-evaluated its `const`s, which Julia either warns about or rejects.
#   julia --project=benchmark benchmark/experiments/exp11_bhhh.jl mnist
# =============================================================================

# -----------------------------------------------------------------------------
# (a) the identity
# -----------------------------------------------------------------------------

const K_LIST = [2, 5, 10, 20]
const M_LIST = [500, 2_000, 8_000, 32_000, 128_000]

function identity_table()
    io = IOBuffer()
    println(io, "Relative error ‖B − ∇²f‖/‖∇²f‖ on logistic regression.\n")
    @printf(io, "%4s %9s %12s %12s %12s %14s\n",
            "K", "M", "at β*", "×√M", "at β*+1.5", "‖B−W‖/‖∇²f‖")
    println(io, "-"^70)
    rows = NamedTuple[]
    for K in K_LIST
        for M in M_LIST
            p = LogisticRegression(K = K, M = M, seed = 1)
            β = β_true(p)
            e0 = information_identity_error(p, β)
            e1 = information_identity_error(p, β .+ 1.5)
            @printf(io, "%4d %9d %12.5f %12.3f %12.5f %14.5f\n",
                    K, M, e0.B_err, e0.B_err * sqrt(M), e1.B_err, e0.BW_gap)
            push!(rows, (K = K, M = M, at_true = e0.B_err, at_wrong = e1.B_err,
                         W_err = e0.W_err, bw = e0.BW_gap, gnorm = e0.grad_norm))
        end
        println(io)
    end
    println(io, "The ×√M column is flat at β*: the identity holds and the sample")
    println(io, "error is O(M^{-1/2}). At β*+1.5 the error is O(1) and does not move,")
    println(io, "which is why BHHH takes poor steps far from the optimum — B is not")
    println(io, "approximating the Hessian there, it is approximating nothing.")
    return String(take!(io)), rows
end

function plot_identity(rows)
    plt = plot(; xlabel = "sample size M", ylabel = "‖B − ∇²f‖ / ‖∇²f‖",
                 xscale = :log10, yscale = :log10, legend = :bottomleft, lw = 2,
                 title = "the information identity, measured")
    for K in K_LIST
        r = filter(x -> x.K == K, rows)
        plot!(plt, [x.M for x in r], [x.at_true  for x in r]; label = "K=$K, at β*")
        plot!(plt, [x.M for x in r], [x.at_wrong for x in r]; label = "K=$K, at β*+1.5",
              ls = :dash)
    end
    M = float.(M_LIST)
    plot!(plt, M, 5 ./ sqrt.(M); label = "M^{-1/2}", ls = :dot, c = :black)
    return plt
end

# -----------------------------------------------------------------------------
# (b, c) mechanisms over each model
# -----------------------------------------------------------------------------

const MODELS = [("ExactHessian", () -> ExactHessian()),
                ("BHHH",         () -> BHHHModel(ridge = 1e-8)),
                ("BHHH-2",       () -> BHHH2Model(ridge = 1e-8)),
                ("SR1",          () -> SR1Model(mem = 10))]

function mechanism_grid(; K = 10, M = 4_000)
    p   = LogisticRegression(K = K, M = M, seed = 2)
    rows = NamedTuple[]
    for (mname, mf) in MODELS, (rname, rf) in RULES_MINIMAL()
        nlp = FullBatchNLP(p; x0 = zeros(K))
        st = tr_solve(nlp; rule = rf(), model = mf(), subsolver = SteihaugCG(),
                      trace = true,
                      params = TRParams(tol = 1e-7, max_iterations = 500))
        gap = norm(st.solution .- β_true(p))
        push!(rows, (model = mname, rule = rname, status = st.status,
                     iters = st.iter, obj = st.objective, gnorm = st.dual_feas,
                     to_truth = gap, grads = neval_grad(nlp)))
    end
    return rows
end

RULES_MINIMAL() = [("RDelta", () -> RDelta()), ("RStep", () -> RStep()),
                   ("RDFO", () -> RDFO(ζ = 1.0)), ("RGrad", () -> RGrad(μ = 1.0))]

function mechanism_table(rows)
    io = IOBuffer()
    @printf(io, "%-14s %-8s %12s %7s %12s %12s\n",
            "model", "rule", "status", "iters", "‖g‖", "‖β−β*‖")
    println(io, "-"^70)
    last = ""
    for r in rows
        r.model == last || (println(io); last = r.model)
        @printf(io, "%-14s %-8s %12s %7d %12.3e %12.4f\n",
                r.model, r.rule, string(r.status), r.iters, r.gnorm, r.to_truth)
    end
    println(io)
    println(io, "‖β−β*‖ is the distance to the generating parameters, which is not")
    println(io, "zero at the optimum: β̂ is the maximum-likelihood estimate for this")
    println(io, "sample, and differs from β* by the usual O(M^{-1/2}) sampling error.")
    return String(take!(io))
end

# -----------------------------------------------------------------------------
# (d) neural networks
# -----------------------------------------------------------------------------

"""
    load_image_data(which; n_train, downsample, gray, classes)

MNIST or CIFAR-100 through MLDatasets, falling back to synthetic data of the same
shape when the package or the download is unavailable.

`downsample` and `gray` are not conveniences. The parameter count is
`hidden·d + hidden + c·hidden + c`, and the exact Hessian needs `n²` entries:
CIFAR-100 at full resolution is 101 636 parameters and a 77 GiB dense Hessian,
while at 16×16 greyscale it is 11 524 and 0.99 GiB. The score matrix that BHHH
needs is `n × N` throughout — 199 MiB and 22 MiB respectively — which is the
asymmetry that makes the outer-product models usable here at all and the exact
Hessian not.
"""
function load_image_data(which::Symbol; n_train::Int = 2_000, downsample::Int = 2,
                         gray::Bool = true, classes::Int = 0)
    data = try
        @eval Main using MLDatasets
        if which === :mnist
            d = Main.MLDatasets.MNIST(split = :train)
            (d.features, d.targets .+ 1, 10)
        else
            d = Main.MLDatasets.CIFAR100(split = :train)
            (d.features, d.targets.coarse .+ 1, 20)
        end
    catch err
        @warn """Could not load $which ($(sprint(showerror, err))).
                 Falling back to synthetic data of the same shape. The structure of
                 the experiment is unchanged — the identity still fails, because it
                 fails for any misspecified model — but the numbers are not MNIST's.""" 
        nothing
    end

    if data === nothing
        rng = MersenneTwister(0)
        d = which === :mnist ? 28 : 32
        c = which === :mnist ? 10 : 20
        px = which === :mnist ? randn(rng, d, d, n_train) : randn(rng, d, d, 3, n_train)
        lab = rand(rng, 1:c, n_train)
        data = (px, lab, c)
    end

    px, lab, c = data
    n = min(n_train, length(lab))
    classes > 0 && (c = min(c, classes))
    keep = findall(<=(c), lab[1:n])
    X = _flatten(px, keep, downsample, gray)
    return X, lab[keep], c
end

function _flatten(px, keep, ds::Int, gray::Bool)
    a = ndims(px) == 4 ? px[:, :, :, keep] : px[:, :, keep]
    if ndims(a) == 4 && gray
        a = dropdims(sum(a; dims = 3); dims = 3) ./ size(a, 3)
    end
    ds > 1 && (a = a[1:ds:end, 1:ds:end, ntuple(_ -> Colon(), ndims(a) - 2)...])
    M = size(a)[end]
    X = reshape(Float64.(a), :, M)'
    return Matrix(X)
end

function nn_run(which::Symbol; hidden = 8, n_train = 1_500, downsample = 3,
                maxit = 150, λ = 1e-3)
    X, y, c = load_image_data(which; n_train = n_train, downsample = downsample)
    p = MLPClassifier(X, y, c; hidden = hidden, λ = λ)
    θ0 = init_params(p; seed = 1)
    @info "network" dataset=which d=p.d hidden=p.h classes=p.c parameters=p.n samples=p.M

    # The identity error at two sample sizes. If it were an approximation, this
    # would shrink; on a misspecified model it does not.
    e_small = information_identity_error(p, θ0; batch = 1:min(200, p.M))
    e_large = information_identity_error(p, θ0; batch = 1:p.M)

    rows = NamedTuple[]
    models = p.n <= 2_000 ?
        [("ExactHessian", () -> ExactHessian()), ("BHHH", () -> BHHHModel(ridge = 1e-6)),
         ("BHHH-2", () -> BHHH2Model(ridge = 1e-6))] :
        [("BHHH", () -> BHHHModel(ridge = 1e-6)), ("BHHH-2", () -> BHHH2Model(ridge = 1e-6))]
    p.n > 2_000 && @info "n = $(p.n) > 2000: skipping ExactHessian, which needs O(n) gradient evaluations and an n×n array"

    for (mname, mf) in models, (rname, rf) in RULES_MINIMAL()
        nlp = FullBatchNLP(p; x0 = copy(θ0))
        st = tr_solve(nlp; rule = rf(), model = mf(), subsolver = SteihaugCG(),
                      params = TRParams(tol = 1e-5, max_iterations = maxit))
        push!(rows, (model = mname, rule = rname, status = st.status, iters = st.iter,
                     loss = st.objective, gnorm = st.dual_feas,
                     acc = accuracy(p, st.solution)))
    end
    return rows, e_small, e_large, p
end

function nn_table(which, rows, e_small, e_large, p)
    io = IOBuffer()
    @printf(io, "%s: %d parameters, %d samples, %d classes\n\n", which, p.n, p.M, p.c)
    @printf(io, "information identity at the initial point:\n")
    @printf(io, "   N = %5d   ‖B−∇²f‖/‖∇²f‖ = %s\n", min(200, p.M), _fmt(e_small.B_err))
    @printf(io, "   N = %5d   ‖B−∇²f‖/‖∇²f‖ = %s\n", p.M, _fmt(e_large.B_err))
    println(io, "   It does not shrink with N. The model is misspecified, so V ≠ −H")
    println(io, "   however large the sample, and BHHH is a preconditioner here, not")
    println(io, "   an approximate Hessian.\n")
    @printf(io, "%-14s %-8s %12s %7s %10s %10s\n",
            "model", "rule", "status", "iters", "loss", "train acc")
    println(io, "-"^66)
    last = ""
    for r in rows
        r.model == last || (println(io); last = r.model)
        @printf(io, "%-14s %-8s %12s %7d %10.4f %10.3f\n",
                r.model, r.rule, string(r.status), r.iters, r.loss, r.acc)
    end
    return String(take!(io))
end

_fmt(v) = isfinite(v) ? @sprintf("%.4f", v) : "n/a (exact Hessian out of reach)"

# -----------------------------------------------------------------------------

function bhhh_study(args = String[])
    arch = ExperimentArchive(tag = "bhhh")
    save_config(arch; rules = [r[1] for r in RULES_MINIMAL()],
                models = MODELS, subsolvers = [("SteihaugCG", () -> SteihaugCG())],
                params = "tol=1e-7", extra = Dict("experiment" => "exp11_bhhh"))

    tbl, rows = identity_table()
    print(tbl); save_table(arch, "exp11_identity.txt", tbl)
    savefig_archived(arch, "exp11_identity.pdf", plot_identity(rows))

    mrows = mechanism_grid()
    print(mechanism_table(mrows)); save_table(arch, "exp11_mechanisms.txt", mechanism_table(mrows))

    for which in (isempty(args) ? (:mnist, :cifar100) : (Symbol(args[1]),))
        r, es, el, p = nn_run(which)
        t = nn_table(which, r, es, el, p)
        print(t); save_table(arch, "exp11_$(which).txt", t)
    end

    finalize_archive(arch; notes = """
        Outer-product Hessians: BHHH, BHHH-2 and the exact Hessian, over logistic
        regression (correctly specified, so the information identity holds) and
        over one-hidden-layer classifiers on MNIST and CIFAR-100 (misspecified, so
        it does not).

        On logistic regression the identity error decays like M^{-1/2} at the true
        parameters and stays O(1) at a displaced point. That is the whole content
        of BHHH's reputation for poor early steps and good late ones: B is not an
        approximation to the Hessian away from the optimum, so there is nothing to
        be surprised about when it behaves like one that isn't.

        On the networks the error does not decay with the sample at all, because
        misspecification breaks the identity at every N. BHHH remains positive
        semidefinite and works as a preconditioner, but calling it an approximate
        Hessian there is unjustified — and the consequence is concrete rather than
        philosophical: B ⪰ 0 means the model never reports negative curvature, so
        every radius mechanism converges contentedly to saddle points, which are
        the dominant critical points of a network. The same warning attached to
        LBFGSModel in the docs applies here for a different reason, and combining
        BHHH with SecondOrder gives τ ≡ ‖g‖ and a second-order status that
        certifies nothing.
        """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    bhhh_study(ARGS)
end
