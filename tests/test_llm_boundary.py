"""Invariants 4 and 5: agent-side code reaches models only through LLMGateway.

Scans agent-side packages for imports of raw providers or LLM SDKs, access to the
gateway's private provider, and unmasking (which belongs in the final human render).
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "aml"
AGENT_SIDE = [SRC / "agents", SRC / "tools", SRC / "validate", SRC / "graph.py"]

FORBIDDEN = re.compile(
    r"\baml\.llm\.providers\b"
    r"|^\s*(?:import|from)\s+(?:anthropic|openai|langchain_aws|langchain_openai)\b"
    r"|\._provider\b"
    r"|\bunmask\s*\(",
    re.MULTILINE,
)


def find_violations(paths: list[Path], root: Path) -> list[str]:
    files = [f for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*.py")))]
    hits = []
    for f in files:
        for lineno, line in enumerate(f.read_text().splitlines(), start=1):
            if m := FORBIDDEN.search(line):
                hits.append(f"{f.relative_to(root)}:{lineno}: {m.group(0).strip()}")
    return hits


def test_agent_side_code_uses_only_the_gateway() -> None:
    assert find_violations(AGENT_SIDE, SRC) == []


def test_scanner_catches_bypasses(tmp_path: Path) -> None:
    bad = tmp_path / "agent.py"
    bad.write_text(
        "import anthropic\n"
        "from aml.llm.providers.fake import FakeLLM\n"
        "gw._provider.complete(p, s)\n"
        "text = masker.unmask(out)\n"
        "from aml.llm.gateway import LLMGateway  # fine\n"
    )
    assert len(find_violations([bad], tmp_path)) == 4
