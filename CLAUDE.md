# AML Agents — project context for Claude Code

## What this is
An agentic anti-money-laundering (AML) system that generates suspicious-activity
alerts, triages them, investigates entities, drafts SAR-style narratives where
every claim cites transaction IDs, and routes cases to a reviewer and a human
sign-off. Portfolio project; built on IBM's synthetic AML dataset (HI-Small).

## Core principle
Deterministic code detects and computes. The LLM prioritizes, explains, and writes.
The LLM never sums, counts, or traverses the transaction graph itself; it only
receives pre-computed evidence objects from tools.

## Non-negotiable invariants
1. **No label leakage.** The `is_laundering` column and the patterns file are used
   ONLY by `src/aml/eval/`. Alert generation, features, tools, and agents must
   never read them. A test enforces this.
2. **Temporal split.** Thresholds and anomaly models are fit on the train period
   only. Evaluation runs on the test period.
3. **Grounded narratives.** Narratives are generated as structured claims
   (`Claim` schema) first, then rendered to prose. Every claim lists `txn_ids`
   and `asserted_fields`; the validator checks them against the database.
4. **No raw PII reaches an LLM.** All entity data passes through the PII
   tokenizer before any prompt is built. Unmasking happens only in the final
   human-facing render. A test scans prompt logs for raw PII.
5. **Everything is audited.** Every LLM call, tool call, and human decision writes
   an event to the hash-chained audit log.
6. **Tests never hit the network.** Use `FakeLLM` in tests. Real providers are
   used only in `scripts/` and the eval harness.

## Stack
- Python 3.11, `uv` for env and deps, `ruff` + `pytest`
- DuckDB for transactions and alerts; SQLite for the audit log and LangGraph checkpoints
- LangGraph for orchestration; pydantic v2 for all schemas
- networkx for graph features; scikit-learn (IsolationForest) for anomaly scoring
- Faker for synthetic customer profiles
- LLM providers behind one interface: `fake`, `anthropic`, `bedrock` (via `langchain-aws` ChatBedrockConverse), `openai`. Model IDs live in config, never hardcoded.

## Layout
```
src/aml/
  config.py          # settings (pydantic-settings, .env)
  data/              # ingest, txn_id minting, synthetic customers, splits
  alerts/            # rules, features, anomaly scoring, alert records
  tools/             # deterministic investigation tools -> Evidence objects
  pii/               # tokenizer + vault
  llm/               # provider interface, FakeLLM, cost/latency accounting
  audit/             # hash-chained event log
  agents/            # triage, investigation, narrative, reviewer nodes
  graph.py           # LangGraph wiring + human interrupt
  validate/          # citation validator
  eval/              # metrics; the ONLY module allowed to read labels
scripts/             # CLI entry points
tests/
data/raw/            # gitignored
```

## Commands
- `uv sync` — install
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format .` — lint/format
- `uv run python -m scripts.<name>` — run a pipeline step

## Conventions
- Small, typed functions; pydantic models at every module boundary.
- Each phase ends with passing tests and a short entry in `docs/PROGRESS.md`
  (what was built, key numbers, open issues).
- Prefer clarity over cleverness; this repo will be read by interviewers.
