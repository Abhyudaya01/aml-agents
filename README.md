# AML Agents

An agentic anti-money-laundering (AML) system, built on IBM's synthetic HI-Small dataset. It generates suspicious-activity alerts, and is designed to triage them, investigate entities, draft SAR-style narratives in which every claim cites transaction IDs, and route cases to a reviewer and a human sign-off.

**Core principle:** deterministic code detects and computes; the LLM prioritizes, explains and writes. The LLM never sums, counts or traverses the transaction graph itself. It only receives pre-computed evidence.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Repo scaffold, settings, smoke test | done |
| 1 | Ingest HI-Small into DuckDB, `txn_id` minting, FX normalization, customers, train/test split | done |
| 2 | Account-window alerts: features, rules, anomaly score, tuning, evaluation | done |
| 3 | LLM gateway, PII tokenizer, hash-chained audit log | done |
| 4+ | Agents (triage, investigation, narrative, reviewer), citation validator, LangGraph wiring with human sign-off | not started |

No agent exists yet; Phases 1–3 are the data, detection and safety infrastructure the agents will build on. Per-phase details, numbers and open issues are in [docs/PROGRESS.md](docs/PROGRESS.md).

## Baseline results

Alert unit is an (account, 7-day window). Thresholds were tuned on the train week only (Sep 1–7), with a budget of 10 alerts per 1,000 train account-windows. The test period starts Sep 8. A test alert is a true positive if its window contains at least one laundering transaction involving that account.

| variant | alerts | precision | recall |
|---|---:|---:|---:|
| combined (rules + anomaly) | 1,386 | 10.4% | 5.4% |
| rules only | 1,366 | 10.5% | 5.4% |
| anomaly only | 2,505 | 3.2% | 3.0% |

This is a deliberately simple baseline, not a good detector. About 1% of windows are positive, so 10% precision is roughly 10× the base rate, but recall is low. Single-account rules catch fan-in and scatter-gather patterns and mostly miss multi-hop ones (cycles, bipartite, random). The IsolationForest adds almost nothing on top of the rules. Per-typology recall and the full analysis are in PROGRESS.md.

## How it works

```
 HI-Small CSV ─► ingest ─► DuckDB: transactions, customers (synthetic), fx_rates
                              │
            eval-only schema: │ txn_labels, laundering_attempts   (never read by detection code)
                              ▼
        legs ─► 7-day account windows ─► features ─► rules + IsolationForest ─► alerts
                                                           ▲
                       artifacts/thresholds.json ◄── tuning on train labels (aml.eval only)

 agent code ─► LLMGateway ─► mask PII ─► raw-PII check ─► provider ─► audit log (hash-chained)
```

### Invariants

These are enforced by tests, not just convention:

1. **No label leakage.** `is_laundering` and the patterns file are read only by `src/aml/eval/`. Labels live in a separate DuckDB schema (`eval`). A scanner test fails if any other module references them.
2. **Temporal split.** Train is before 2022-09-08, test is on or after it. Windows are anchored on the split date so none mixes both. Thresholds, the anomaly model and its score percentiles are fit on train only; tests check that flipping test labels changes nothing.
3. **Grounded narratives** *(planned)*: claims will be generated as structured objects citing `txn_ids`, then validated against the database.
4. **No raw PII reaches an LLM.** Entity data is replaced with stable tokens (`CUST_0042`, `ACCT_0913`) before any prompt is built, and a final check blocks prompts containing known raw values. Unmasking is for the final human-facing render only.
5. **Everything is audited.** Every LLM call (including blocked and failed ones) writes an event to an append-only log where each row's hash covers the previous row's. `verify_chain()` detects edits, deletions and reordering.
6. **Tests never hit the network.** They use `FakeLLM`. Real providers are for scripts and the eval harness.

### Components

| Path | Purpose |
|---|---|
| `src/aml/data/` | Ingest, `txn_id` minting, FX table, synthetic customers (Faker, seeded), train/test split |
| `src/aml/alerts/` | Legs and windows, features, rules, anomaly scoring, alert records |
| `src/aml/eval/` | The only package allowed to read labels: label tables, pattern matching, tuning, alert metrics |
| `src/aml/llm/` | Provider interface (`fake`, `anthropic`, `bedrock`, `openai`), structured output with validation retry, cost accounting, `LLMGateway` |
| `src/aml/pii/` | Reversible token vault (SQLite), `mask` / `unmask`, raw-PII detector |
| `src/aml/audit/` | Hash-chained event log |
| `src/aml/tools/`, `agents/`, `validate/`, `graph.py` | Reserved for later phases |
| `scripts/` | CLI entry points: `ingest`, `alerts` |

Design notes worth knowing:

- **The source has no transaction ID.** `txn_id = sha256(raw_line | row_index)[:16]`; the row index breaks ties between exact-duplicate rows.
- **Accounts are keyed by (bank, account),** because 8 account IDs appear under two banks.
- **Currency conversion uses rates derived from the dataset's own cross-currency rows,** so converted amounts agree with the simulator (median error under 0.001%). Original and USD amounts are both stored.
- **Alert rules** are FAN_IN, FAN_OUT, STRUCTURING, VELOCITY_SPIKE and PASS_THROUGH. Each is one SQL condition plus an evidence query that returns the triggering `txn_id`s. The tuner currently leaves FAN_OUT, STRUCTURING and VELOCITY_SPIKE off.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Data

Download IBM's *Transactions for Anti Money Laundering (AML)* dataset (e.g. from Kaggle) and place two files in `data/raw/` (gitignored):

- `HI-Small_Trans.csv` (about 475 MB, 5.08M rows)
- `HI-Small_Patterns.txt`

### Run the pipeline

```bash
uv run python -m scripts.ingest   # ~90 s; builds data/aml.duckdb, reports into docs/PROGRESS.md
uv run python -m scripts.alerts   # ~15 s; tunes on train, writes artifacts/thresholds.json, evaluates test
```

Use `uv run python -m scripts.alerts --frozen` to re-run with the committed thresholds without re-tuning.

### Configuration

Copy `.env.example` to `.env`. All settings use the `AML_` prefix: data paths, split date, alert budget, and the LLM provider (`fake` by default). The default model for `anthropic` is `claude-opus-5-5`; for `openai` and `bedrock`, set `AML_MODEL_ID` (and `AML_AWS_REGION` for Bedrock). Built-in prices cover Anthropic models only; add others with `AML_LLM_PRICE_OVERRIDES`, since a model without a price is rejected rather than logged at zero cost.

## Development

```bash
uv run pytest                              # 111 tests, no network, ~12 s
uv run ruff check . && uv run ruff format .
```

The tests build tiny hand-made datasets, so they run without the HI-Small files. The full-data pipeline is not part of the test suite.

## Limitations

- The real LLM providers (Anthropic, OpenAI, Bedrock) have only been tested against stub clients, not live APIs.
- The detection baseline is weak on multi-hop typologies; graph features are the planned next step.
- 38% of laundering rows (1,968 of 5,177) are not in any typed attempt in the patterns file, so per-typology recall covers only the rest.
- The test period's laundering rate is twice train's, partly because of a sparse Sep 11–18 tail in the simulator. Interpret absolute metrics with care.
- Python cannot fully prevent a deliberate bypass of the LLM gateway at runtime; the boundary relies on structure, a static scan test and the gateway's own PII check.
- Data is synthetic; nothing here has been validated against real transaction monitoring.
