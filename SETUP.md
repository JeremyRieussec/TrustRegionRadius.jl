# Finishing the package

Concrete steps, in order. Everything before "Deploying to GitHub" is local and takes
about ten minutes; the rest is one-time repository configuration.

---

## 1. Set the UUID

`Project.toml` ships with a placeholder. A real one has been generated for you:

```
7a8d5bce-8bd3-4f8e-8a33-850f3a7a2263
```

It must appear in **three** places, identically:

| file | line |
|---|---|
| `Project.toml` | `uuid = "7a8d5bce-…"` |
| `benchmark/Project.toml` | `TrustRegionRadius = "7a8d5bce-…"` |
| `docs/Project.toml` | `TrustRegionRadius = "7a8d5bce-…"` |

If you would rather generate your own:

```julia
using UUIDs; uuid4()
```

A mismatch between the three does not error at `using` time — it fails later, when
`Pkg.instantiate()` in the benchmark or docs environment cannot resolve the dependency.
Worth getting right now.

---

## 2. Instantiate and check that it loads

```bash
cd TrustRegionRadius
julia --project -e 'using Pkg; Pkg.instantiate(); using TrustRegionRadius'
```

The first `using` is the real test of everything written so far. Expect to touch two things:

- **`LinearOperators` constructor signatures.** `src/Model_Hessians/model_hessian.jl` calls
  `LBFGSOperator(Float64, n, mem = m)` and `SR1Operator(Float64, n, mem = m)`. Some releases
  use `LBFGSOperator(n; mem = m)`. Check with `?LBFGSOperator`.
- **`set_solver_specific!`.** `src/Trust-region/solver.jl` uses it under `trace = true`. Its
  signature has moved between SolverCore releases; if it errors, `?set_solver_specific!`.

---

## 3. Run the tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The suite is deliberately opinionated: it asserts not just that the code runs but that the
survey's claims hold — that uncapped `RGrad` drives μ past any threshold, that `RStep`'s
`Δmin` prevents the collapse to zero, that `cg_step_info` returns `cos(s,-g) = 1` on a tiny
radius, that `SPDTarget` throws exactly when φ ≥ 0. A failure there is a finding, not just a
bug.

---

## 4. Instantiate the two auxiliary environments

They are separate so that Plots, JLD2 and CUTEst are not dependencies of the package itself.

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs      -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

CUTEst needs a working SIF decoder and compiler toolchain. If it fails to build, the
benchmark harness falls back to `analytic_problems()` and everything else still runs — that
fallback is deliberate, so a missing CUTEst never blocks the rest of the campaign.

---

## 5. Build the documentation locally

```bash
julia --project=docs docs/make.jl
```

Open `docs/build/index.html`. `Documenter` will warn about any exported symbol whose
docstring is missing or malformed; treat those warnings as a checklist. All 47 exports are
already listed in `docs/src/index.md`, so a warning means a docstring problem, not a missing
entry.

---

## Deploying to GitHub

### 6. Initialise and push

```bash
git init
git add .
git commit -m "Initial commit: trust-region radius update mechanisms"
git branch -M main
git remote add origin https://github.com/JeremyRieussec/TrustRegionRadius.jl.git
git push -u origin main
```

Two conventions worth following, because tooling depends on them:

- **The repository must end in `.jl`** — `TrustRegionRadius.jl`. Documenter, TagBot and the
  General registry all assume it.
- **`Manifest.toml` stays untracked.** It is already in `.gitignore`. A package pins
  `[compat]` ranges, not exact versions; committing a Manifest makes the package
  unusable in any environment that resolves differently.

### 7. Continuous integration

Copy the three workflow files from `.github/workflows/` (supplied alongside this document).

| file | what it does |
|---|---|
| `CI.yml` | runs the test suite on Julia LTS + stable, on Linux/macOS/Windows |
| `Documenter.yml` | builds and publishes the docs to `gh-pages` on every push to `main` |
| `CompatHelper.yml` | opens PRs when a dependency's `[compat]` bound falls behind |
| `TagBot.yml` | creates a GitHub release when a version is registered |

The CI matrix deliberately excludes the benchmark environment: CUTEst cannot be built on a
runner in reasonable time, and the analytic fallback would test the harness rather than the
package.

### 8. Documentation hosting

`docs/make.jl` currently has `deploydocs` commented out. Uncomment and set the repository:

```julia
deploydocs(
    repo      = "github.com/JeremyRieussec/TrustRegionRadius.jl.git",
    devbranch = "main",
)
```

Then authorise the build to push:

```julia
using Pkg; Pkg.add("DocumenterTools")
using DocumenterTools
DocumenterTools.genkeys(user = "JeremyRieussec", repo = "TrustRegionRadius.jl")
```

This prints two keys. Add the **public** one as a deploy key named `documenter` with write
access (Settings → Deploy keys), and the **private** one as a repository secret named
`DOCUMENTER_KEY` (Settings → Secrets and variables → Actions).

Finally, in Settings → Pages, set the source to the `gh-pages` branch. The docs appear at
`https://JeremyRieussec.github.io/TrustRegionRadius.jl/`.

### 9. Registration (optional)

Only if you want `Pkg.add("TrustRegionRadius")` to work for others. The package must have an
OSI licence, semantic versioning, and tests that pass on CI.

Install the [Julia Registrator](https://github.com/JuliaRegistries/Registrator.jl) GitHub App,
then comment on the commit you want released:

```
@JuliaRegistrator register
```

TagBot creates the git tag and GitHub release once the registry PR merges. For a package
that exists mainly to support a paper, registration is optional — a `Pkg.develop(url=...)`
in the README is often enough, and avoids committing to the registry's version discipline.

---

## 10. Make the paper reproducible

Since this package backs Parts I–III, two extra steps are worth the effort.

**Archive a DOI.** Link the repository to [Zenodo](https://zenodo.org), then create a GitHub
release. Zenodo mints a DOI for that exact snapshot, and the badge goes in the README and the
paper's reproducibility statement. This is what lets a referee run the code you actually ran.

**Record the environment with the results.** Each experiment archive already writes
`experiment_config.toml`. Add the resolved versions to it by extending `save_config`:

```julia
extra = Dict("julia_version" => string(VERSION),
             "manifest"      => read(joinpath(dirname(Base.active_project()),
                                              "Manifest.toml"), String))
```

The config file is what makes an archive self-describing; without the versions, a run that
cannot be reproduced two years from now cannot be diagnosed either.

---

## Checklist

- [x] UUID set identically in `Project.toml`, `benchmark/Project.toml`, `docs/Project.toml`
- [x] `using TrustRegionRadius` succeeds
- [x] `Pkg.test()` passes
- [x] `LinearOperators` and `set_solver_specific!` signatures confirmed
- [x] benchmark and docs environments instantiated
- [x] `docs/make.jl` builds locally with no warnings
- [x] `LICENSE` present (MIT supplied; change if your institution requires otherwise)
- [x] repository named `TrustRegionRadius.jl`
- [x] `Manifest.toml` untracked
- [ ] workflows copied, `DOCUMENTER_KEY` set, Pages pointed at `gh-pages`
- [ ] `deploydocs` uncommented with the real repository
- [ ] CI green
- [ ] Zenodo DOI minted from a release
