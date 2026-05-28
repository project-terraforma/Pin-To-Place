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
# Longer / more-specific keys take priority in prefix matching.
_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o-mini":              (0.00015,  0.0006),
    "gpt-4o":                   (0.0025,   0.010),
    "gpt-4-turbo":              (0.010,    0.030),
    # Anthropic — list full date-suffixed IDs first so prefix match is unambiguous
    "claude-haiku-4-5-20251001": (0.00025, 0.00125),
    "claude-haiku-4-5":          (0.00025, 0.00125),
    "claude-sonnet-4-6":         (0.003,   0.015),
    "claude-opus-4-7":           (0.015,   0.075),
}

_PROVIDER_PREFIXES: dict[str, str] = {
    "gpt-":    "OpenAI",
    "claude-": "Anthropic",
    "o1":      "OpenAI",
    "o3":      "OpenAI",
}


def _provider(model: str) -> str:
    for prefix, name in _PROVIDER_PREFIXES.items():
        if model.startswith(prefix):
            return name
    return "Unknown"


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    # Sort by key length descending so the most specific match wins
    key = next(
        (k for k in sorted(_PRICING, key=len, reverse=True) if model.startswith(k)),
        None,
    )
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
    """Record one API call. Returns the cost in USD for this call."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cost = _cost(model, input_tokens, output_tokens)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_label": run_label,
        "place_id": place_id,
        "model": model,
        "provider": _provider(model),
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

    total_input  = sum(r["input_tokens"]  for r in records)
    total_output = sum(r["output_tokens"] for r in records)
    total_cost   = sum(r["cost_usd"]      for r in records)
    total_calls  = len(records)

    # ── By provider ───────────────────────────────────────────────────────────
    by_provider: dict[str, dict] = {}
    for r in records:
        prov = r.get("provider") or _provider(r["model"])
        if prov not in by_provider:
            by_provider[prov] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        by_provider[prov]["calls"]         += 1
        by_provider[prov]["input_tokens"]  += r["input_tokens"]
        by_provider[prov]["output_tokens"] += r["output_tokens"]
        by_provider[prov]["cost_usd"]      += r["cost_usd"]

    # ── By model ──────────────────────────────────────────────────────────────
    by_model: dict[str, dict] = {}
    for r in records:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        by_model[m]["calls"]         += 1
        by_model[m]["input_tokens"]  += r["input_tokens"]
        by_model[m]["output_tokens"] += r["output_tokens"]
        by_model[m]["cost_usd"]      += r["cost_usd"]

    # ── By run label ──────────────────────────────────────────────────────────
    by_label: dict[str, dict] = {}
    for r in records:
        lbl = r["run_label"]
        if lbl not in by_label:
            by_label[lbl] = {"calls": 0, "cost_usd": 0.0}
        by_label[lbl]["calls"]    += 1
        by_label[lbl]["cost_usd"] += r["cost_usd"]

    # ── Projected cost for full 3425-place run ────────────────────────────────
    projection_line = ""
    if total_calls > 0:
        cost_per_call = total_cost / total_calls
        # Each place = 2 primary calls (dual-label Tier 1/2) + ~0.015 CV overhead
        projected = cost_per_call * 2 * 3425
        projection_line = f"  Projected full run (3425 places): ~${projected:.2f}"

    lines = [
        "Pin-To-Place — LLM Usage Summary",
        f"Last updated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Log entries  : {total_calls:,}",
        "",
        "── Totals ──────────────────────────────────────────",
        f"  Input tokens  : {total_input:>12,}",
        f"  Output tokens : {total_output:>12,}",
        f"  Total tokens  : {total_input + total_output:>12,}",
        f"  Total cost    : ${total_cost:>11.4f}",
        "",
    ]

    if projection_line:
        lines += [projection_line, ""]

    lines += ["── By Provider ─────────────────────────────────────"]
    for prov, stats in sorted(by_provider.items()):
        lines.append(
            f"  {prov:<12}  {stats['calls']:>5} calls  "
            f"{stats['input_tokens'] + stats['output_tokens']:>10,} tokens  "
            f"${stats['cost_usd']:.4f}"
        )

    lines += ["", "── By Model ────────────────────────────────────────"]
    for m, stats in sorted(by_model.items()):
        lines.append(f"  {m}")
        lines.append(f"    calls   : {stats['calls']:>6,}")
        lines.append(f"    tokens  : {stats['input_tokens'] + stats['output_tokens']:>10,}")
        lines.append(f"    cost    : ${stats['cost_usd']:.4f}")

    lines += ["", "── By Run Label ────────────────────────────────────"]
    for label, stats in sorted(by_label.items()):
        lines.append(
            f"  {label:<35}  {stats['calls']:>4} calls   ${stats['cost_usd']:.4f}"
        )

    (LOGS_DIR / "usage_summary.txt").write_text("\n".join(lines) + "\n")


def print_summary() -> None:
    summary_path = LOGS_DIR / "usage_summary.txt"
    if summary_path.exists():
        print(summary_path.read_text())
    else:
        print("No usage recorded yet.")
