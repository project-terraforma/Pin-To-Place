"""
Token usage and cost tracking for LLM API calls.
Appends one JSON record per call to logs/usage_log.jsonl and rewrites
logs/usage_summary.txt with running totals after every update.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# Prices in USD per 1,000 tokens (input, output)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":            (0.0025, 0.010),
    "gpt-4o-mini":       (0.00015, 0.0006),
    "gpt-4-turbo":       (0.010, 0.030),
    "claude-opus-4-7":   (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-haiku-4-5":  (0.00025, 0.00125),
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    key = next((k for k in sorted(_PRICING, key=len, reverse=True) if model.startswith(k)), None)
    if key is None:
        return 0.0
    price_in, price_out = _PRICING[key]
    return round(input_tokens * price_in / 1000 + output_tokens * price_out / 1000, 6)


def log_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    run_label: str = "unknown",
    place_id: str = "",
) -> float:
    """
    Record one API call. Returns the cost in USD for this call.
    Creates logs/ directory if it doesn't exist.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cost = _cost(model, input_tokens, output_tokens)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "place_id": place_id,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
    }

    with open(LOGS_DIR / "usage_log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")

    _rewrite_summary()
    return cost


def _rewrite_summary() -> None:
    log_path = LOGS_DIR / "usage_log.jsonl"
    if not log_path.exists():
        return

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not records:
        return

    total_input = sum(r["input_tokens"] for r in records)
    total_output = sum(r["output_tokens"] for r in records)
    total_cost = sum(r["cost_usd"] for r in records)

    # Per-model breakdown
    by_model: dict[str, dict] = {}
    for r in records:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        by_model[m]["calls"] += 1
        by_model[m]["input_tokens"] += r["input_tokens"]
        by_model[m]["output_tokens"] += r["output_tokens"]
        by_model[m]["cost_usd"] += r["cost_usd"]

    # Per-run-label breakdown
    by_label: dict[str, dict] = {}
    for r in records:
        lbl = r["run_label"]
        if lbl not in by_label:
            by_label[lbl] = {"calls": 0, "cost_usd": 0.0}
        by_label[lbl]["calls"] += 1
        by_label[lbl]["cost_usd"] += r["cost_usd"]

    lines = [
        "Pin-To-Place — LLM Usage Summary",
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Log entries:  {len(records)}",
        "",
        "── Totals ──────────────────────────────",
        f"  Input tokens:   {total_input:>10,}",
        f"  Output tokens:  {total_output:>10,}",
        f"  Total tokens:   {total_input + total_output:>10,}",
        f"  Total cost:     ${total_cost:>10.4f}",
        "",
        "── By Model ────────────────────────────",
    ]
    for model, stats in sorted(by_model.items()):
        lines.append(f"  {model}")
        lines.append(f"    calls:  {stats['calls']:>6}")
        lines.append(f"    tokens: {stats['input_tokens'] + stats['output_tokens']:>6,}")
        lines.append(f"    cost:   ${stats['cost_usd']:.4f}")

    lines += ["", "── By Run Label ────────────────────────"]
    for label, stats in sorted(by_label.items()):
        lines.append(f"  {label:<30}  {stats['calls']:>4} calls   ${stats['cost_usd']:.4f}")

    (LOGS_DIR / "usage_summary.txt").write_text("\n".join(lines) + "\n")


def print_summary() -> None:
    summary_path = LOGS_DIR / "usage_summary.txt"
    if summary_path.exists():
        print(summary_path.read_text())
    else:
        print("No usage recorded yet.")
