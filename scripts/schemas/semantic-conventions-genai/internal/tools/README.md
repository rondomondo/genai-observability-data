# internal/tools

Maintainer-only scripts. Not part of the public contributor workflow — none
of these are wired into `make` targets or `CONTRIBUTING.md`. Run them
directly with `uv` from the repo root.

## `split_model_md.py`

Review-aid that rewrites generated cross-links in `docs/` so they target the
per-namespace pages (`gen-ai.md` / `mcp.md` / `openai.md`) instead of the
single merged `model.md` Weaver emits for a `definition/2` registry. Reduces
diff noise; does not affect the generation pipeline itself.

```sh
make generate-registry
uv run internal/tools/split_model_md.py
```

Idempotent: rerunning is a no-op once no `model.md` references remain.
