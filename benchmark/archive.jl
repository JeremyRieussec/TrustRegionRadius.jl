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
    :η₁ => "eta1",   :η₂ => "eta2",   :η  => "eta",
    :η̃₁ => "eta1_t", :η̃₂ => "eta2_t",
    :γ₀ => "gamma0", :γ₁ => "gamma1", :γ₂ => "gamma2", :γ₃ => "gamma3",
    :Δ₀ => "Delta0", :Δmax => "Delta_max", :Δmin => "Delta_min",
    :ζ  => "zeta",   :μ  => "mu",     :μ₀ => "mu0",    :μ_max => "mu_max",
    :β  => "beta",   :λ₁ => "lambda1", :λ₂ => "lambda2", :M => "M",
    :max_iterations => "max_iterations", :tol => "tol", :max_time => "max_time",
    :half_test => "half_test",
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
    struct_to_dict(obj) -> Dict{String, Any}

Introspect any struct into an ASCII-keyed dictionary. `Inf` and `NaN` become
strings, since TOML has no representation for them.
"""
function struct_to_dict(obj)::Dict{String, Any}
    d = Dict{String, Any}()
    d["type"] = string(nameof(typeof(obj)))
    for f in fieldnames(typeof(obj))
        d[field_to_ascii(f)] = _toml_value(getfield(obj, f))
    end
    return d
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
    save_config(archive; rules, params, problem_selection = Dict(), extra = Dict())

Write `experiment_config.toml` into the archive.

`rules` may be a vector of `(name, factory)` pairs, of `(name, rule)` pairs, or
of bare rules. Factories are called once to capture the *initial* parameter
values, which is what should be recorded: a mutable rule's μ after a run says
nothing about how the run was configured.
"""
function save_config(a::ExperimentArchive;
                     rules = [],
                     params = nothing,
                     problem_selection::Dict = Dict{String, Any}(),
                     extra::Dict = Dict{String, Any}())
    cfg = Dict{String, Any}()
    cfg["generated_at"] = string(a.created)
    isempty(a.tag) || (cfg["tag"] = a.tag)

    params === nothing || (cfg["solver_params"] = struct_to_dict(params))
    isempty(problem_selection) ||
        (cfg["problem_selection"] = Dict{String, Any}(
            string(k) => _toml_value(v) for (k, v) in problem_selection))

    rule_dicts = Dict{String, Any}[]
    for entry in rules
        name, obj = if entry isa Tuple || entry isa Pair
            n, r = entry
            (String(n), r isa Function ? r() : r)
        else
            (string(nameof(typeof(entry))), entry)
        end
        d = struct_to_dict(obj)
        d["name"] = name
        push!(rule_dicts, d)
    end
    isempty(rule_dicts) || (cfg["rules"] = rule_dicts)

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

    if haskey(cfg, "rules")
        println(io, "## Radius Update Rules\n")
        for r in cfg["rules"]
            fields = join(("$k = $(r[k])" for k in sort(collect(keys(r)))
                           if k ∉ ("name", "type")), ", ")
            println(io, "- **", get(r, "name", "?"), "** (`", get(r, "type", "?"),
                    "`): ", fields)
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
