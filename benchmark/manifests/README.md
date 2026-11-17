# Frozen manifests

This directory keeps the dependency manifests of the campaigns as records. Julia never
reads a manifest from here, so no Pkg command can rewrite one. The live `Manifest.toml`
files are ignored by git, and each machine resolves its own.

## `v0.1.0/`

The environment of the seven campaigns archived at
[10.5281/zenodo.22655838](https://doi.org/10.5281/zenodo.22655838), resolved under Julia
1.12.1 on the machine that ran them.

| file | belongs at | bytes | content hash |
|:--|:--|--:|:--|
| `Manifest-root.toml` | `Manifest.toml` | 4347 | `b5e9c5e29c995cb9` |
| `Manifest-benchmark.toml` | `benchmark/Manifest.toml` | 60315 | `1076809e89f89806` |

The hash is the one `provenance()` computes in `benchmark/archive.jl`:

```julia
string(hash(read("benchmark/manifests/v0.1.0/Manifest-root.toml", String)); base = 16)
```

Every archive records `manifest_hash = "b5e9c5e29c995cb9"`, the hash of the root manifest,
so `Manifest-root.toml` is checkable against the archives. The `benchmark` manifest, the
environment actually active during the runs, was never hashed by a campaign; its hash is
given here so that a later copy can be compared with this one.

Both files are stored byte for byte. `.gitattributes` switches off line-ending conversion
for this directory, because a checkout that turned LF into CRLF would change the hash.

## Using them

To rerun a campaign in its own environment, copy the two files to where Julia looks and
instantiate without resolving. `benchmark/REPRODUCE.md` gives the full procedure.

```bash
cp benchmark/manifests/v0.1.0/Manifest-root.toml Manifest.toml
cp benchmark/manifests/v0.1.0/Manifest-benchmark.toml benchmark/Manifest.toml
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
```

Use Julia 1.12 for this, the minor version the manifests were resolved for. Under another
one the standard-library versions they record do not match, and resolving again leaves the
campaign environment.

## Adding one

When a new release produces archived campaigns, add a directory named for its tag with the
two manifests of the machine that ran them, and a row in a table like the one above. Never
edit a file already here.
