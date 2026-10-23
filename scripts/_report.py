"""Replace a generated, marker-delimited block in docs/PROGRESS.md."""

from pathlib import Path

PROGRESS_PATH = Path("docs/PROGRESS.md")


def markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}-report:start -->", f"<!-- {name}-report:end -->"


def write_block(name: str, body: str, path: Path = PROGRESS_PATH) -> None:
    start, end = markers(name)
    block = f"{start}\n{body.strip()}\n{end}"
    text = path.read_text()
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = head + block + tail
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    path.write_text(text)
