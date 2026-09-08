# =============================================================================
# benchmark/archive.jl
#
# Experiment archiving.
#
# Every run creates one self-documenting directory:
#
#   results/
#     exp_2026-04-16_02-08-34/
#       experiment_config.toml     what was run
#       experiment_summary.md      what came out
#       figures/                   every PDF produced
#       tables/                    every text table
#       data/                      raw JLD2 results
#
# The directory name carries the timestamp so runs never overwrite one another
# and are ordered lexicographically by date.
# =============================================================================

using TOML
using Dates
using JLD2

# -----------------------------------------------------------------------------
# Unicode → ASCII field names for TOML
# -----------------------------------------------------------------------------

"""
    UNICODE_TO_ASCII :: Dict{Symbol, String}

Field-name translation for TOML keys. TOML has no objection to Unicode, but
ASCII keys survive `grep`, spreadsheet imports and copy-paste into a paper
without surprises.

Unmapped symbols fall through to `string(sym)`.
"""
const UNICODE_TO_ASCII = Dict{Symbol, String}(
    :η1 => "eta1",   :η2 => "eta2",   :η  => "eta",
    :η̃₁ => "eta1_t", :η̃₂ => "eta2_t",
    :γ0 => "gamma0", :γ1 => "gamma1", :γ2 => "gamma2", :γ3 => "gamma3",
    :Δ0 => "Delta0", :Δmax => "Delta_max", :Δmin => "Delta_min",
    :ζ  => "zeta",   :μ  => "mu",     :μ0 => "mu0",    :μ_max => "mu_max",
    :β  => "beta",   :λ1 => "lambda1", :λ2 => "lambda2", :M => "M",
    :max_iterations => "max_iterations", :tol => "tol", :max_time => "max_time",
    :half_test => "half_test",
    :χ  => "chi",    :θ  => "theta",
)

field_to_ascii(sym::Symbol) = get(UNICODE_TO_ASCII, sym, string(sym))

"TOML accepts only a few scalar types; coerce everything else to a string."
function _toml_value(v)
    v isa Union{Bool, Integer, AbstractString} && return v
    v isa Real && return isfinite(v) ? Float64(v) : string(v)   # Inf/NaN are not TOML
    v isa AbstractVector && return [_toml_value(x) for x in v]
    return string(v)
end

"""
    _is_component(v) -> Bool

True when a field holds one of the three configured axes in its own right.

Such a field is recorded as a nested table rather than flattened by
`_toml_value`, so `EigenPoint(SteihaugCG())` records the inner solver's `χ`, `θ`
and `max_iters` instead of the single string `"SteihaugCG(0.1, 0.0001, 1000)"`,
and a `SecondOrder` wrapper records the rule it wraps.
"""
_is_component(v) = v isa Union{RadiusRule, ModelHessian, SubproblemSolver}

"""
    struct_to_dict(obj) -> Dict{String, Any}

Introspect any struct into an ASCII-keyed dictionary. `Inf` and `NaN` become
strings, since TOML has no representation for them, and a field holding another
rule, model or subsolver becomes a nested table.
"""
function struct_to_dict(obj)::Dict{String, Any}
    d = Dict{String, Any}()
    d["type"] = string(nameof(typeof(obj)))
    for f in fieldnames(typeof(obj))
        v = getfield(obj, f)
        d[field_to_ascii(f)] = _is_component(v) ? struct_to_dict(v) : _toml_value(v)
    end
    return d
end

"""
    _component_dicts(entries) -> Vector{Dict{String, Any}}

Normalise a list describing one configured axis into TOML tables.

An entry may be a `(name, factory)` pair, a `(name, instance)` pair, a bare
factory, a bare instance, or a bare name. Factories are called once to capture
the *initial* parameter values, which is what should be recorded: a mutable
rule's μ after a run says nothing about how the run was configured.

A bare name records only that name. That is weaker than an instance, but it is
honest about being weaker: the previous code sent a bare name down the struct
path and recorded `type = "String"` for every entry, so an archive of three
named rules read as three identical anonymous ones.
"""
function _component_dicts(entries)::Vector{Dict{String, Any}}
    out = Dict{String, Any}[]
    for entry in entries
        name, obj = if entry isa Tuple || entry isa Pair
            n, r = entry
            (String(n), r isa Function ? r() : r)
        elseif entry isa AbstractString
            (String(entry), nothing)
        elseif entry isa Function
            o = entry()
            (string(nameof(typeof(o))), o)
        else
            (string(nameof(typeof(entry))), entry)
        end
        d = obj === nothing ? Dict{String, Any}("type" => name) : struct_to_dict(obj)
        d["name"] = name
        push!(out, d)
    end
    return out
end

"Deduplicate component tables by content, keeping first-seen order."
function _unique_components(ds::Vector{Dict{String, Any}})
    out = Dict{String, Any}[]
    for d in ds
        any(isequal(d), out) || push!(out, d)
    end
    return out
end

# -----------------------------------------------------------------------------
# ExperimentArchive
# -----------------------------------------------------------------------------

"""
    ExperimentArchive(root = "results"; tag = "", timestamp = now())

Create a timestamped experiment directory and return a handle to it.

The directory is `\$root/exp_YYYY-MM-DD_HH-MM-SS[_tag]/`, containing `figures/`,
`tables/` and `data/`. Pass a `tag` to distinguish concurrent runs
(`exp_2026-04-16_02-08-34_zeta_sweep`).

```julia
arch = ExperimentArchive()
save_config(arch; rules = RULES, params = SOLVER_PARAMS)
savefig_archived(arch, "perf_profile.pdf", plt)
save_table(arch, "success_rate.txt", table_string)
finalize_archive(arch)          # writes experiment_summary.md
```
"""
struct ExperimentArchive
    root::String
    dir::String
    figures::String
    tables::String
    data::String
    created::DateTime
    tag::String
    meta::Dict{String, Any}
end

function ExperimentArchive(root::AbstractString = joinpath(@__DIR__, "results");
                           tag::AbstractString = "",
                           timestamp::DateTime = now(),
                           resume::Union{Nothing,AbstractString} = nothing)
    # Reprise : soit l'argument `resume`, soit la variable d'environnement
    # TRR_RESUME, ce qui permet de relancer un script d'expérience inchangé.
    #     TRR_RESUME=results/exp_...  julia --project=benchmark .../exp1.jl
    dir_resume = resume === nothing ? get(ENV, "TRR_RESUME", "") : String(resume)
    if !isempty(dir_resume)
        return reopen_archive(dir_resume)
    end
    return _new_archive(root, tag, timestamp)
end

function _new_archive(root::AbstractString, tag::AbstractString, timestamp::DateTime)
    stamp = Dates.format(timestamp, "yyyy-mm-dd_HH-MM-SS")
    name  = isempty(tag) ? "exp_$stamp" : "exp_$(stamp)_$tag"
    dir   = joinpath(root, name)
    figs  = joinpath(dir, "figures")
    tabs  = joinpath(dir, "tables")
    data  = joinpath(dir, "data")
    for d in (dir, figs, tabs, data)
        mkpath(d)
    end
    @info "Experiment archive → $dir"
    return ExperimentArchive(String(root), dir, figs, tabs, data,
                             timestamp, String(tag), Dict{String, Any}())
end

Base.show(io::IO, a::ExperimentArchive) =
    print(io, "ExperimentArchive(", basename(a.dir), ")")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

"""
    _repo_root() -> String

The package root, found from this file rather than from the active project, so
that provenance is recorded even when the benchmark is run from its own
environment.
"""
_repo_root() = normpath(joinpath(@__DIR__, ".."))

"""
    _content_hash(path) -> String

A short content hash of a file, or `""` when it is missing.

`hash` from `Base`, not a cryptographic digest: this exists to detect that a
`Manifest.toml` changed between two campaigns, not to defend against anyone
forging one, and it avoids adding a dependency to record a dependency.
"""
function _content_hash(path::AbstractString)
    isfile(path) || return ""
    return string(hash(read(path, String)); base = 16)
end

"""
    _git_state(root) -> (commit, dirty)

The current commit and whether the tree is clean, or `("", false)` when `git` is
unavailable or the directory is not a repository. Never throws: provenance that
can fail a campaign is worse than provenance that is absent.
"""
function _git_state(root::AbstractString)
    commit = try
        readchomp(setenv(`git rev-parse HEAD`; dir = root))
    catch
        return ("", false)
    end
    dirty = try
        !isempty(readchomp(setenv(`git status --porcelain`; dir = root)))
    catch
        false
    end
    return (commit, dirty)
end

"""
    _package_version(root) -> String

The `version` field of the package's `Project.toml`, or `""`.
"""
function _package_version(root::AbstractString)
    path = joinpath(root, "Project.toml")
    isfile(path) || return ""
    try
        return string(get(TOML.parsefile(path), "version", ""))
    catch
        return ""
    end
end

"""
    provenance(; seed = nothing) -> Dict{String, Any}

What is needed to reproduce a campaign, beyond the parameters themselves: the
Julia version, the package version and commit, whether the tree was dirty, a
content hash of both `Project.toml` and `Manifest.toml`, and the master seed.

Written into every `experiment_config.toml` automatically. A stochastic table
whose seed is not recorded cannot be reproduced, only resampled, and a
dependency set that is not pinned makes a rerun a different experiment.

`git_dirty = true` is not a warning to be silenced: it says the recorded commit
does not describe the code that ran.
"""
function provenance(; seed = nothing)
    root = _repo_root()
    commit, dirty = _git_state(root)
    d = Dict{String, Any}(
        "julia_version"   => string(VERSION),
        "package_version" => _package_version(root),
        "git_commit"      => commit,
        "git_dirty"       => dirty,
        "project_hash"    => _content_hash(joinpath(root, "Project.toml")),
        "manifest_hash"   => _content_hash(joinpath(root, "Manifest.toml")),
        "recorded_at"     => string(Dates.now()),
        "nthreads"        => Threads.nthreads(),
    )
    seed === nothing || (d["seed"] = seed)
    return d
end

"""
    save_config(archive; rules, models, subsolvers, configs, params,
                problem_selection = Dict(), extra = Dict())

Write `experiment_config.toml` into the archive.

A run is a triple: a radius rule, a model Hessian and a subproblem solver. Only
the first was recorded before, which left the archive unable to answer the two
questions a reader of the paper asks first — was that `RGradCapped` row solved
with an exact Hessian or a quasi-Newton one, and was the step a truncated-CG one
or an exact Moré-Sorensen one. `models` and `subsolvers` record the other two.

Each of `rules`, `models` and `subsolvers` accepts `(name, factory)` pairs,
`(name, instance)` pairs, bare factories, bare instances, or bare names; see
[`_component_dicts`](@ref).

`configs` is stronger, and is what the experiments that build a grid should
pass: a vector of `(name, factory)` where the factory returns the
`(rule, model, subsolver)` named tuple that `run_experiment` consumes. It
records the combinations actually solved rather than three independent lists,
which is the difference between experiment 6's 4×4 cross and experiment 1's
eight rules against one fixed pair. Any of the three axis lists left empty is
then derived from `configs`.
"""
function save_config(a::ExperimentArchive;
                     rules = [],
                     models = [],
                     subsolvers = [],
                     configs = [],
                     params = nothing,
                     problem_selection::Dict = Dict{String, Any}(),
                     seed = nothing,
                     extra::Dict = Dict{String, Any}())
    cfg = Dict{String, Any}()
    cfg["generated_at"] = string(a.created)
    # Always, and not through `extra`: provenance that is optional is provenance
    # that is missing from the one campaign whose numbers end up in the paper.
    cfg["provenance"] = provenance(seed = seed)
    isempty(a.tag) || (cfg["tag"] = a.tag)

    params === nothing || (cfg["solver_params"] = struct_to_dict(params))
    isempty(problem_selection) ||
        (cfg["problem_selection"] = Dict{String, Any}(
            string(k) => _toml_value(v) for (k, v) in problem_selection))

    rule_dicts      = _component_dicts(rules)
    model_dicts     = _component_dicts(models)
    subsolver_dicts = _component_dicts(subsolvers)

    config_dicts = Dict{String, Any}[]
    for entry in configs
        name, obj = if entry isa Tuple || entry isa Pair
            n, f = entry
            (String(n), f isa Function ? f() : f)
        else
            ("", entry isa Function ? entry() : entry)
        end
        d = Dict{String, Any}("name" => name)
        for k in (:rule, :model, :subsolver)
            hasproperty(obj, k) || continue
            d[string(k)] = struct_to_dict(getproperty(obj, k))
        end
        push!(config_dicts, d)
    end

    # Derive an axis only when the caller did not state it, and copy before
    # naming: the deduplicated table is the same object that sits inside
    # `configurations`, and adding a `name` key there would edit the cross.
    for (key, dicts) in (("rule", rule_dicts), ("model", model_dicts),
                         ("subsolver", subsolver_dicts))
        isempty(dicts) || continue
        derived = Dict{String, Any}[d[key] for d in config_dicts if haskey(d, key)]
        for d in _unique_components(derived)
            e = copy(d)
            haskey(e, "name") || (e["name"] = get(e, "type", "?"))
            push!(dicts, e)
        end
    end

    isempty(rule_dicts)      || (cfg["rules"]          = rule_dicts)
    isempty(model_dicts)     || (cfg["models"]         = model_dicts)
    isempty(subsolver_dicts) || (cfg["subsolvers"]     = subsolver_dicts)
    isempty(config_dicts)    || (cfg["configurations"] = config_dicts)

    for (k, v) in extra
        cfg[string(k)] = v isa Dict ? v : _toml_value(v)
    end

    path = joinpath(a.dir, "experiment_config.toml")
    open(path, "w") do io
        TOML.print(io, cfg; sorted = true)
    end
    a.meta["config"] = cfg
    @info "Wrote config → $path"
    return path
end

"""
    load_config(dir) -> Dict

Read back the `experiment_config.toml` of an archived run, so a past experiment
can be reproduced or its parameters quoted in a paper.
"""
function load_config(dir::AbstractString)
    path = isdir(dir) ? joinpath(dir, "experiment_config.toml") : String(dir)
    isfile(path) || error("No experiment_config.toml at $path")
    return TOML.parsefile(path)
end

# -----------------------------------------------------------------------------
# Figures, tables, data
# -----------------------------------------------------------------------------

"""
    savefig_archived(archive, filename, plt) -> String

Save a figure into `figures/` and record it for the summary. `filename` should
carry its own extension; `.pdf` is the sensible default for a paper, since it
stays vector.
"""
function savefig_archived(a::ExperimentArchive, filename::AbstractString, plt)
    path = joinpath(a.figures, String(filename))
    savefig(plt, path)
    @info "figure → $(joinpath("figures", filename))"
    return path
end

"""
    save_table(archive, filename, content) -> String

Write a text table into `tables/`. `content` may be a `String` or anything
`show`-able; a `Matrix` is written with `writedlm`-style tab separation.
"""
function save_table(a::ExperimentArchive, filename::AbstractString, content)
    path = joinpath(a.tables, String(filename))
    open(path, "w") do io
        content isa AbstractString ? print(io, content) : show(io, MIME"text/plain"(), content)
    end
    @info "table → $(joinpath("tables", filename))"
    return path
end

"""
    reopen_archive(dir) -> ExperimentArchive

Rouvrir une archive existante au lieu d'en créer une nouvelle.

Les sous-répertoires manquants sont recréés, de sorte qu'une archive
interrompue avant la première figure reste utilisable. La configuration déjà
écrite est relue dans `meta`, ce qui permet à `finalize_archive` de reproduire
le résumé complet.

```julia
arch = reopen_archive("benchmark/results/exp_2026-07-29_02-15-23_comparison")
```

Ou, sans modifier le script d'expérience :

```bash
TRR_RESUME=benchmark/results/exp_2026-07-29_02-15-23_comparison \
  julia --project=benchmark benchmark/experiments/exp1_comparison.jl
```
"""
function reopen_archive(dir::AbstractString)
    isdir(dir) || error("reopen_archive : répertoire introuvable : $dir")
    figs = joinpath(dir, "figures")
    tabs = joinpath(dir, "tables")
    data = joinpath(dir, "data")
    for d in (figs, tabs, data)
        mkpath(d)
    end

    name = basename(rstrip(dir, ['/', '\\']))
    m = match(r"^exp_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_(.*))?$", name)
    created = m === nothing ? now() :
              DateTime(replace(m.captures[1], "_" => "T"), dateformat"yyyy-mm-ddTHH-MM-SS")
    tag = (m === nothing || m.captures[2] === nothing) ? "" : m.captures[2]

    meta = Dict{String, Any}()
    cfgpath = joinpath(dir, "experiment_config.toml")
    isfile(cfgpath) && (meta["config"] = TOML.parsefile(cfgpath))

    n = length(filter(f -> endswith(f, ".jld2"), readdir(data)))
    @info "Archive rouverte → $dir  ($n fichier(s) de données déjà présent(s))"
    return ExperimentArchive(dirname(rstrip(dir, ['/', '\\'])), dir,
                             figs, tabs, data, created, tag, meta)
end

"""
    data_filename(problem, config) -> String

Nom de fichier canonique d'une exécution, utilisé aussi bien à l'écriture qu'à
la relecture. Les caractères hostiles aux systèmes de fichiers sont remplacés.

C'est indispensable : les configurations de l'expérience 6 s'appellent
`"RDelta/exact"`, et une barre oblique ferait écrire dans un sous-répertoire
inexistant.
"""
function data_filename(problem::AbstractString, config::AbstractString)
    clean(x) = replace(String(x), r"[/\\:*?\"<>|]" => "-")
    return string(clean(problem), "__", clean(config), ".jld2")
end

"""
    has_data(archive, problem, config) -> Bool

Vrai si l'exécution correspondante a déjà été enregistrée.
"""
has_data(a::ExperimentArchive, problem, config) =
    isfile(joinpath(a.data, data_filename(problem, config)))

"""
    load_data(archive, problem, config) -> Dict{String,Any}

Relire un fichier de données. Lève une exception si le fichier est absent ou
illisible ; l'appelant doit traiter ce cas comme « à recalculer », puisqu'une
écriture interrompue laisse un JLD2 tronqué.
"""
function load_data(a::ExperimentArchive, problem, config)
    return JLD2.load(joinpath(a.data, data_filename(problem, config)))
end

"""
    save_data(archive, filename; kwargs...) -> String

Write raw results into `data/` as JLD2. Every keyword becomes a stored key.

```julia
save_data(arch, "GENROSE_RDelta.jld2";
          problem_name = "GENROSE", rule_name = "RDelta",
          status = stats.status, iterations = stats.iter)
```
"""
function save_data(a::ExperimentArchive, filename::AbstractString; kwargs...)
    # Assainir : un nom de configuration peut contenir « / » (expérience 6).
    safe = replace(String(filename), r"[/\\:*?\"<>|]" => "-")
    path = joinpath(a.data, safe)
    jldsave(path; kwargs...)
    return path
end

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

"""
    _describe_component(d) -> String

One-line `field = value` rendering of a component table. A nested table (an
`EigenPoint`'s inner solver, a `SecondOrder`'s rule) renders inline as
`type(fields)` rather than as a dictionary literal.
"""
function _describe_component(d::AbstractDict)
    parts = String[]
    for k in sort(collect(keys(d)))
        k in ("name", "type") && continue
        v = d[k]
        push!(parts, v isa AbstractDict ?
              string(k, " = ", get(v, "type", "?"), "(", _describe_component(v), ")") :
              string(k, " = ", v))
    end
    return join(parts, ", ")
end

"`type(fields)`, or bare `type` for a component with no parameters."
function _component_label(d::AbstractDict)
    t = string(get(d, "type", "?"))
    desc = _describe_component(d)
    return isempty(desc) ? t : string(t, "(", desc, ")")
end

"""
    finalize_archive(archive; notes = "") -> String

Write `experiment_summary.md`: the configuration in tables, the inventory of
figures, tables and data files, and any `notes`.

Call once at the end of a run. Re-calling regenerates the file, so it is safe
to call after adding more figures.
"""
function finalize_archive(a::ExperimentArchive; notes::AbstractString = "")
    cfg = get(a.meta, "config", Dict{String, Any}())
    figs = sort(readdir(a.figures))
    tabs = sort(readdir(a.tables))
    dats = sort(readdir(a.data))

    io = IOBuffer()
    println(io, "# Experiment Summary\n")
    println(io, "**Archive:** `", basename(a.dir), "`  ")
    println(io, "**Generated:** ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "\n")
    isempty(notes) || println(io, notes, "\n")

    if haskey(cfg, "solver_params")
        println(io, "## Solver Parameters\n")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        for k in sort(collect(keys(cfg["solver_params"])))
            k == "type" && continue
            println(io, "| `", k, "` | ", cfg["solver_params"][k], " |")
        end
        println(io)
    end

    if haskey(cfg, "problem_selection")
        println(io, "## Problem Selection\n")
        println(io, "| Criterion | Value |")
        println(io, "|-----------|-------|")
        for k in sort(collect(keys(cfg["problem_selection"])))
            println(io, "| `", k, "` | ", cfg["problem_selection"][k], " |")
        end
        println(io)
    end

    for (title, key) in (("Radius Update Rules", "rules"),
                         ("Model Hessians",      "models"),
                         ("Subproblem Solvers",  "subsolvers"))
        haskey(cfg, key) || continue
        println(io, "## ", title, "\n")
        for r in cfg[key]
            # No trailing colon for a component with no parameters: `ExactHessian`
            # has none, and "ExactHessian (`ExactHessian`): " reads as truncated.
            desc = _describe_component(r)
            println(io, "- **", get(r, "name", "?"), "** (`", get(r, "type", "?"),
                    isempty(desc) ? "`)" : string("`): ", desc))
        end
        println(io)
    end

    # The cross, when one was run. Three separate axis lists cannot distinguish
    # experiment 6, which crosses four rules with four models, from experiment 1,
    # which runs eight rules against one fixed model and subsolver, and a summary
    # that reads the same for both is not a record of what happened.
    if haskey(cfg, "configurations")
        println(io, "## Configurations\n")
        println(io, "| Configuration | Rule | Model Hessian | Subsolver |")
        println(io, "|---------------|------|---------------|-----------|")
        for c in cfg["configurations"]
            cell(k) = haskey(c, k) ? _component_label(c[k]) : "--"
            println(io, "| ", get(c, "name", "?"), " | ", cell("rule"), " | ",
                    cell("model"), " | ", cell("subsolver"), " |")
        end
        println(io)
    end

    println(io, "## Outputs\n")
    println(io, "| Item | Count |")
    println(io, "|------|-------|")
    println(io, "| Figures | ", length(figs), " |")
    println(io, "| Tables  | ", length(tabs), " |")
    println(io, "| Data files | ", length(dats), " |")
    println(io)

    for (title, items, sub) in (("Figures", figs, "figures"),
                                ("Tables", tabs, "tables"),
                                ("Data", dats, "data"))
        isempty(items) && continue
        println(io, "### ", title, "\n")
        for f in items
            println(io, "- `", sub, "/", f, "`")
        end
        println(io)
    end

    path = joinpath(a.dir, "experiment_summary.md")
    write(path, String(take!(io)))
    @info "Wrote summary → $path"
    return path
end

"""
    latest_archive(root = "results") -> String

Path of the most recent archive under `root`. Convenient for post-processing
the run that just finished without retyping its timestamp.
"""
function latest_archive(root::AbstractString = joinpath(@__DIR__, "results"))
    isdir(root) || error("No results directory at $root")
    dirs = filter(d -> startswith(d, "exp_") && isdir(joinpath(root, d)), readdir(root))
    isempty(dirs) && error("No experiment archives in $root")
    return joinpath(root, sort(dirs)[end])
end
