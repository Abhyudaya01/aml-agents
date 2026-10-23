# AML Agents

An agentic anti-money-laundering system built on IBM's synthetic HI-Small dataset. Deterministic code detects and computes; the LLM prioritizes, explains, and writes SAR-style narratives in which every claim cites transaction IDs. Cases are routed to a reviewer and a human sign-off.

## Quickstart

```bash
uv sync
uv run pytest
uv run ruff check .
```

Copy `.env.example` to `.env` to override settings. See [CLAUDE.md](CLAUDE.md) for architecture and invariants, and [docs/PROGRESS.md](docs/PROGRESS.md) for build status.
