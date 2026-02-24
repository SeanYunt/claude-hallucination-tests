"""
Pytest configuration and shared fixtures for Claude adversarial testing.

Set ANTHROPIC_API_KEY in your environment or a .env file before running.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class ProbeResult:
    """Captures a single prompt/response pair with metadata."""
    category: str
    prompt: str
    response: str
    model: str
    flagged: bool = False
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def log(self) -> None:
        flag = "⚑ FLAGGED" if self.flagged else "  ok"
        logger.info("[%s] %s | prompt=%.80s...", self.category, flag, self.prompt)
        logger.debug("Response: %s", self.response[:500])

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "flagged": self.flagged,
            "notes": self.notes,
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "timestamp": self.timestamp,
        }


class ResultsCollector:
    """Accumulates ProbeResults and writes a JSON report at session end."""

    def __init__(self) -> None:
        self.results: list[ProbeResult] = []

    def add(self, result: ProbeResult) -> ProbeResult:
        result.log()
        self.results.append(result)
        return result

    def write_report(self, path: Path) -> None:
        data = {
            "generated": datetime.now(UTC).isoformat(),
            "total": len(self.results),
            "flagged": sum(1 for r in self.results if r.flagged),
            "results": [r.as_dict() for r in self.results],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Report written to %s (%d results, %d flagged)",
                    path, data["total"], data["flagged"])


@pytest.fixture(scope="session")
def api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set — skipping all live API tests")
    return key


@pytest.fixture(scope="session")
def client(api_key: str) -> Anthropic:
    return Anthropic(api_key=api_key)


@pytest.fixture(scope="session")
def collector() -> ResultsCollector:
    return ResultsCollector()


@pytest.fixture(scope="session", autouse=True)
def write_report_on_exit(collector: ResultsCollector) -> None:  # noqa: PT004
    yield
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    collector.write_report(RESULTS_DIR / f"run_{ts}.json")


def probe(
    client: Anthropic,
    collector: ResultsCollector,
    *,
    category: str,
    prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    system: str | None = None,
    max_tokens: int = 512,
) -> ProbeResult:
    """
    Send a single prompt to Claude and capture the result.
    Does NOT hard-fail — every response is recorded for human review.
    """
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    text = response.content[0].text if response.content else ""

    result = ProbeResult(category=category, prompt=prompt, response=text, model=model)
    return collector.add(result)


def multi_probe(
    client: Anthropic,
    collector: ResultsCollector,
    *,
    category: str,
    messages: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    system: str | None = None,
    max_tokens: int = 512,
) -> ProbeResult:
    """
    Send a pre-constructed multi-turn conversation and capture the final response.

    `messages` is a list of {"role": ..., "content": ...} dicts representing
    the full conversation history, ending with a user message. This allows
    tests to embed plausible prior turns (e.g. a correct assistant answer)
    before applying adversarial pressure in the final user turn.
    """
    kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    text = response.content[0].text if response.content else ""

    # Summarise the conversation for reporting (truncate each turn for readability)
    prompt_summary = " → ".join(
        f"[{m['role']}]: {str(m['content'])[:80]}" for m in messages
    )
    result = ProbeResult(category=category, prompt=prompt_summary, response=text, model=model)
    return collector.add(result)
