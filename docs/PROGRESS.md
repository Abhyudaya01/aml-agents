# Progress

## Phase 0: scaffold

**Built**
- Repo layout per `CLAUDE.md`: empty `aml.*` packages, `scripts/`, `tests/`, `docs/`.
- `uv` project (Python 3.11, hatchling, `src/` layout), ruff and pytest config.
- `aml.config.Settings` (pydantic-settings, `AML_` prefix): data paths, DuckDB path, audit DB path, `llm_provider`, `model_id`, `aws_region`, train/test split date.
- Smoke test that imports every package.

**Key numbers**: none yet.

**Open issues**
- Train/test split date is a placeholder; set it after inspecting the HI-Small date range.
- Only pydantic and pydantic-settings are declared; other stack dependencies are added in the phase that first uses them.
