#!/usr/bin/env python3
"""
Session Replay Comparison — Validates retro insights by replaying optimal paths.
================================================================================

The core question: Does your retro process produce actionable insights?

This tool takes a completed project session (PLAN.md, HISTORY.jsonl, RETRO.md)
and does three things:

1. EXTRACT — Pulls the "optimal path" from the retro (the sequence of prompts
   that would have reached the same result with minimal waste)

2. REPLAY — Feeds that optimal path through the agent with the same system
   prompt and measures: turns to completion, waste ratio, criteria met

3. COMPARE — Side-by-side comparison of original session vs. replay:
   actual turns vs. optimal turns, waste distribution, where the retro
   insights prevented specific dead-ends

Also supports:
- Manual optimal path definition (you write the replay prompts yourself)
- A/B replay: same optimal path through different system prompt variants
- Partial replay: replay from a specific milestone forward

Usage:
    # Auto-extract optimal path from retro and replay
    python replay.py auto --session-dir ./my-project --output replay-results/

    # Replay a manually defined optimal path
    python replay.py manual --playbook optimal_prompts.jsonl --system-prompt agent-plan-prompt-v2.md --output replay-results/

    # Compare original vs replay
    python replay.py compare --original ./my-project --replay replay-results/

    # A/B: replay same path through two variants
    python replay.py ab --playbook optimal_prompts.jsonl --variants full,minimal --output replay-results/

    # Extract optimal path from retro (without replaying)
    python replay.py extract --session-dir ./my-project --output optimal_prompts.jsonl

Requires: ANTHROPIC_API_KEY environment variable
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Constants ───────────────────────────────────────────────────────────────

PRODUCTIVE_TAGS = {"clean-execution", "good-discovery", "course-correct"}
WASTE_TAGS = {"dead-end", "redo", "yak-shave", "scope-creep", "vague-prompt", "misfire"}
NEUTRAL_TAGS = {"planning", "debug", "refactor"}


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ReplayPrompt:
    """A single prompt in the optimal replay sequence."""
    sequence_num: int
    prompt_text: str
    rationale: str = ""           # Why this prompt is in the optimal path
    original_ids: list = field(default_factory=list)  # Original P-IDs this replaces
    skipped_ids: list = field(default_factory=list)   # Original P-IDs this makes unnecessary
    expected_outcome: str = ""     # What this should accomplish

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplayTurn:
    """Result of a single replay turn."""
    sequence_num: int
    prompt: str
    response: str
    latency_ms: int
    token_count_prompt: int
    token_count_response: int
    process_tag: Optional[str] = None
    has_footer: bool = False

    def to_dict(self) -> dict:
        return {
            "sequence_num": self.sequence_num,
            "prompt": self.prompt[:500],
            "response": self.response[:500],
            "latency_ms": self.latency_ms,
            "token_count_prompt": self.token_count_prompt,
            "token_count_response": self.token_count_response,
            "process_tag": self.process_tag,
            "has_footer": self.has_footer,
        }


@dataclass
class SessionMetrics:
    """Metrics for a session (original or replay)."""
    label: str
    total_turns: int
    productive_turns: int = 0
    waste_turns: int = 0
    neutral_turns: int = 0
    untagged_turns: int = 0
    criteria_met: list = field(default_factory=list)
    criteria_missed: list = field(default_factory=list)
    total_tokens: int = 0
    total_latency_ms: int = 0
    tag_distribution: dict = field(default_factory=dict)

    @property
    def productive_ratio(self) -> float:
        tagged = self.productive_turns + self.waste_turns + self.neutral_turns
        return self.productive_turns / tagged if tagged else 0.0

    @property
    def waste_ratio(self) -> float:
        tagged = self.productive_turns + self.waste_turns + self.neutral_turns
        return self.waste_turns / tagged if tagged else 0.0

    @property
    def completion_rate(self) -> float:
        total = len(self.criteria_met) + len(self.criteria_missed)
        return len(self.criteria_met) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total_turns": self.total_turns,
            "productive_turns": self.productive_turns,
            "waste_turns": self.waste_turns,
            "neutral_turns": self.neutral_turns,
            "productive_ratio": round(self.productive_ratio, 4),
            "waste_ratio": round(self.waste_ratio, 4),
            "completion_rate": round(self.completion_rate, 4),
            "criteria_met": self.criteria_met,
            "criteria_missed": self.criteria_missed,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "tag_distribution": self.tag_distribution,
        }


@dataclass
class ReplayComparison:
    """Side-by-side comparison of original vs replay."""
    original: SessionMetrics
    replay: SessionMetrics
    optimal_path: list[ReplayPrompt] = field(default_factory=list)
    insights_validated: list[str] = field(default_factory=list)
    insights_invalidated: list[str] = field(default_factory=list)
    replay_turns: list[ReplayTurn] = field(default_factory=list)

    @property
    def turn_reduction(self) -> float:
        """How many fewer turns the replay took (negative = replay was worse)."""
        if self.original.total_turns == 0:
            return 0.0
        return 1.0 - (self.replay.total_turns / self.original.total_turns)

    @property
    def waste_reduction(self) -> float:
        """How much waste was eliminated."""
        if self.original.waste_ratio == 0:
            return 0.0
        return 1.0 - (self.replay.waste_ratio / self.original.waste_ratio)

    @property
    def optimal_path_ratio(self) -> float:
        """Replay turns / original turns. <1.0 means retro helped."""
        if self.original.total_turns == 0:
            return 0.0
        return self.replay.total_turns / self.original.total_turns

    def to_dict(self) -> dict:
        return {
            "comparison": {
                "turn_reduction": round(self.turn_reduction, 4),
                "waste_reduction": round(self.waste_reduction, 4),
                "optimal_path_ratio": round(self.optimal_path_ratio, 4),
                "insights_validated": self.insights_validated,
                "insights_invalidated": self.insights_invalidated,
            },
            "original": self.original.to_dict(),
            "replay": self.replay.to_dict(),
            "optimal_path": [p.to_dict() for p in self.optimal_path],
            "replay_turns": [t.to_dict() for t in self.replay_turns],
        }

    def summary(self) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("  SESSION REPLAY COMPARISON")
        lines.append("=" * 65)

        # Side by side
        lines.append(f"\n  {'Metric':<30} {'Original':>12} {'Replay':>12} {'Delta':>10}")
        lines.append(f"  {'-'*64}")

        def row(label, orig, replay, fmt=".0f", invert=False):
            if isinstance(orig, float):
                delta = replay - orig
                if invert:
                    delta = -delta
                d_str = f"{delta:+{fmt}}"
                color = "\033[92m" if (delta < 0 if not invert else delta > 0) else "\033[91m"
                return f"  {label:<30} {orig:>12{fmt}} {replay:>12{fmt}} {color}{d_str:>10}\033[0m"
            return f"  {label:<30} {str(orig):>12} {str(replay):>12}"

        lines.append(row("Total turns", self.original.total_turns, self.replay.total_turns, "d"))
        lines.append(row("Productive turns", self.original.productive_turns, self.replay.productive_turns, "d", invert=True))
        lines.append(row("Waste turns", self.original.waste_turns, self.replay.waste_turns, "d"))
        lines.append(row("Productive %", self.original.productive_ratio * 100, self.replay.productive_ratio * 100, ".0f", invert=True))
        lines.append(row("Waste %", self.original.waste_ratio * 100, self.replay.waste_ratio * 100, ".0f"))
        lines.append(row("Completion %", self.original.completion_rate * 100, self.replay.completion_rate * 100, ".0f", invert=True))
        lines.append(row("Total tokens", self.original.total_tokens, self.replay.total_tokens, ",d"))

        # Key metrics
        lines.append(f"\n  📊 Optimal Path Ratio: {self.optimal_path_ratio:.2f}x")
        lines.append(f"     (1.0 = same length, <1.0 = retro helped, >1.0 = retro hurt)")

        if self.turn_reduction > 0:
            lines.append(f"  ✅ Turn reduction: {self.turn_reduction:.0%} fewer turns")
        elif self.turn_reduction < 0:
            lines.append(f"  ❌ Turn inflation: {-self.turn_reduction:.0%} MORE turns (retro insights may be wrong)")

        if self.waste_reduction > 0:
            lines.append(f"  ✅ Waste reduction: {self.waste_reduction:.0%} less waste")

        # Insights
        if self.insights_validated:
            lines.append(f"\n  ✅ Validated Retro Insights ({len(self.insights_validated)}):")
            for insight in self.insights_validated:
                lines.append(f"     · {insight}")

        if self.insights_invalidated:
            lines.append(f"\n  ❌ Invalidated Insights ({len(self.insights_invalidated)}):")
            for insight in self.insights_invalidated:
                lines.append(f"     · {insight}")

        # Tag distribution comparison
        lines.append(f"\n  Tag Distribution:")
        all_tags = sorted(set(list(self.original.tag_distribution.keys()) + list(self.replay.tag_distribution.keys())))
        for tag in all_tags:
            o = self.original.tag_distribution.get(tag, 0)
            r = self.replay.tag_distribution.get(tag, 0)
            if o > 0 or r > 0:
                lines.append(f"    {tag:<20} {o:>4} → {r:>4}")

        lines.append("\n" + "=" * 65)
        return "\n".join(lines)


# ── Optimal Path Extraction ────────────────────────────────────────────────

EXTRACT_PROMPT = """You are analyzing a completed coding session to extract the optimal prompt sequence.

Given the full session history (prompts, responses, process tags) and the retro analysis, your job is to produce the SHORTEST possible sequence of prompts that would achieve the same result.

Rules:
- Eliminate all waste turns (dead-ends, redos, scope-creep, yak-shaves)
- Merge vague-prompt + correction into a single specific prompt
- Keep planning turns only if they changed the approach
- Keep debug turns only for bugs that were unavoidable
- Preserve the sequencing lessons from the retro's "Optimal Path" section
- Each prompt should be self-contained and specific (input → operation → output)
- Include learnings from the retro to avoid the same dead-ends

SESSION HISTORY:
{history_entries}

RETRO ANALYSIS:
{retro_content}

DONE CRITERIA (from the session goal):
{done_criteria}

Respond with ONLY a JSON array of prompt objects, no markdown:
[
  {{
    "sequence_num": 1,
    "prompt_text": "The exact prompt to send to the agent",
    "rationale": "Why this prompt is needed and what waste it avoids",
    "original_ids": ["P-001", "P-002"],
    "skipped_ids": ["P-003", "P-004"],
    "expected_outcome": "What this turn should accomplish"
  }}
]"""

VALIDATE_PROMPT = """You are comparing the results of a session replay against the original session.

The original session had these retro insights:
{retro_insights}

The replay produced these results:
- Turns: {replay_turns} (original: {original_turns})
- Waste ratio: {replay_waste:.0%} (original: {original_waste:.0%})
- Completion: {replay_completion:.0%} (original: {original_completion:.0%})

Replay tag distribution: {replay_tags}

For each retro insight, determine if it was VALIDATED (the replay avoided the original waste) or INVALIDATED (the same waste occurred anyway, or new waste appeared).

Respond with ONLY this JSON:
{{
  "validated": ["insight 1 text", "insight 2 text"],
  "invalidated": ["insight 3 text"],
  "reasoning": "Brief explanation of what worked and what didn't"
}}"""


class OptimalPathExtractor:
    """Extracts optimal prompt sequence from a completed session."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            print("Install anthropic: pip install anthropic --break-system-packages")
            sys.exit(1)

    def extract(self, session_dir: str, verbose: bool = True) -> list[ReplayPrompt]:
        """Extract optimal path from session artifacts."""
        session_path = Path(session_dir)

        # Load session data
        history_path = session_path / "HISTORY.jsonl"
        retro_path = session_path / "RETRO.md"
        plan_path = session_path / "PLAN.md"

        if not history_path.exists():
            raise FileNotFoundError(f"HISTORY.jsonl not found in {session_dir}")

        entries = []
        for line in history_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Build history summary for the extraction prompt
        history_summary = []
        for e in entries:
            tag = e.get("process_tag", "")
            note = e.get("process_note", "")
            tag_str = f" [{tag}]" if tag else ""
            note_str = f" — {note}" if note else ""
            history_summary.append(
                f"{e['id']}: {e.get('summary', '')}{tag_str}{note_str}"
            )

        retro_content = ""
        if retro_path.exists():
            retro_content = retro_path.read_text()
        else:
            retro_content = "[No retro available — extract optimal path from history alone]"

        # Extract done criteria from PLAN.md goal
        done_criteria = ""
        if plan_path.exists():
            plan_content = plan_path.read_text()
            goal_match = re.search(r'## Goal\s*\n(.*?)(?=\n##|\Z)', plan_content, re.DOTALL)
            if goal_match:
                done_criteria = goal_match.group(1).strip()

        if verbose:
            print(f"  Extracting optimal path from {len(entries)} entries...")

        # Call the extraction model
        prompt = EXTRACT_PROMPT.format(
            history_entries="\n".join(history_summary),
            retro_content=retro_content[:4000],
            done_criteria=done_criteria or "[Infer from session history]",
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            raw_prompts = json.loads(text)

            replay_prompts = []
            for rp in raw_prompts:
                replay_prompts.append(ReplayPrompt(
                    sequence_num=rp.get("sequence_num", len(replay_prompts) + 1),
                    prompt_text=rp.get("prompt_text", ""),
                    rationale=rp.get("rationale", ""),
                    original_ids=rp.get("original_ids", []),
                    skipped_ids=rp.get("skipped_ids", []),
                    expected_outcome=rp.get("expected_outcome", ""),
                ))

            if verbose:
                print(f"  ✓ Extracted {len(replay_prompts)} optimal prompts (from {len(entries)} original entries)")

            return replay_prompts

        except json.JSONDecodeError as e:
            print(f"  ⚠ Failed to parse extraction response: {e}")
            return []
        except Exception as e:
            print(f"  ⚠ Extraction failed: {e}")
            return []


# ── Session Metrics Extraction ──────────────────────────────────────────────

def extract_original_metrics(session_dir: str) -> SessionMetrics:
    """Extract metrics from an original completed session."""
    session_path = Path(session_dir)
    history_path = session_path / "HISTORY.jsonl"

    if not history_path.exists():
        raise FileNotFoundError(f"HISTORY.jsonl not found in {session_dir}")

    entries = []
    for line in history_path.read_text().strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    responses = [e for e in entries if e.get("type") == "response"]
    prompts = [e for e in entries if e.get("type") == "prompt"]

    metrics = SessionMetrics(label="original", total_turns=len(prompts))

    tag_dist = {}
    for r in responses:
        tag = r.get("process_tag", "")
        if tag:
            tag_dist[tag] = tag_dist.get(tag, 0) + 1
            if tag in PRODUCTIVE_TAGS:
                metrics.productive_turns += 1
            elif tag in WASTE_TAGS:
                metrics.waste_turns += 1
            elif tag in NEUTRAL_TAGS:
                metrics.neutral_turns += 1
        else:
            metrics.untagged_turns += 1

    metrics.tag_distribution = tag_dist
    return metrics


# ── Replay Engine ───────────────────────────────────────────────────────────

MINIMAL_SYSTEM_PROMPT = """You are a coding agent. Maintain two files:
PLAN.md — Track what you're building, current status, and next steps.
HISTORY.jsonl — Log each prompt/response with an ID, summary, and tags.
After every turn, update both files and tell the user what you did and what's next."""


class ReplayEngine:
    """Replays an optimal prompt sequence through the agent."""

    def __init__(
        self,
        system_prompt: str = "",
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            print("Install anthropic: pip install anthropic --break-system-packages")
            sys.exit(1)

    def _extract_process_tag(self, response: str) -> Optional[str]:
        match = re.search(r"\*\*Process:?\*\*\s*\[?`?(\S+?)`?\]?\s", response)
        if match:
            return match.group(1).strip("`*[]")
        return None

    def _check_footer(self, response: str) -> bool:
        return bool(re.search(
            r"\*\*Done:?\*\*.*\*\*Next:?\*\*.*\*\*Process:?\*\*",
            response, re.DOTALL,
        ))

    def replay(
        self,
        prompts: list[ReplayPrompt],
        verbose: bool = True,
    ) -> tuple[list[ReplayTurn], SessionMetrics]:
        """Execute the replay sequence and collect metrics."""
        turns = []
        conversation = []
        metrics = SessionMetrics(label="replay", total_turns=0)
        tag_dist = {}

        if verbose:
            print(f"\n  Replaying {len(prompts)} optimal prompts...")

        for rp in prompts:
            user_msg = rp.prompt_text
            conversation.append({"role": "user", "content": user_msg})

            if verbose:
                print(f"\n  [{rp.sequence_num}] {user_msg[:80]}...")
                if rp.rationale:
                    print(f"      Rationale: {rp.rationale[:80]}")

            try:
                start = time.time()
                kwargs = {
                    "model": self.model,
                    "max_tokens": 4096,
                    "messages": conversation,
                }
                if self.system_prompt:
                    kwargs["system"] = self.system_prompt

                response = self.client.messages.create(**kwargs)
                latency = int((time.time() - start) * 1000)
                response_text = response.content[0].text
                conversation.append({"role": "assistant", "content": response_text})

                tag = self._extract_process_tag(response_text)
                turn = ReplayTurn(
                    sequence_num=rp.sequence_num,
                    prompt=user_msg,
                    response=response_text,
                    latency_ms=latency,
                    token_count_prompt=response.usage.input_tokens,
                    token_count_response=response.usage.output_tokens,
                    process_tag=tag,
                    has_footer=self._check_footer(response_text),
                )
                turns.append(turn)

                # Track metrics
                metrics.total_turns += 1
                metrics.total_tokens += response.usage.input_tokens + response.usage.output_tokens
                metrics.total_latency_ms += latency

                if tag:
                    tag_dist[tag] = tag_dist.get(tag, 0) + 1
                    if tag in PRODUCTIVE_TAGS:
                        metrics.productive_turns += 1
                    elif tag in WASTE_TAGS:
                        metrics.waste_turns += 1
                    elif tag in NEUTRAL_TAGS:
                        metrics.neutral_turns += 1

                if verbose:
                    tag_str = f" [{tag}]" if tag else " [no tag]"
                    print(f"    → {latency}ms, {response.usage.output_tokens} tokens{tag_str}")

            except Exception as e:
                if verbose:
                    print(f"    ✗ Error: {e}")
                break

        metrics.tag_distribution = tag_dist
        return turns, metrics

    def evaluate_completion(
        self,
        conversation: list[dict],
        done_criteria: list[str],
    ) -> tuple[list[str], list[str]]:
        """Evaluate which done criteria were met in the replay."""
        if not done_criteria:
            return [], []

        conv_text = "\n\n".join(
            f"{'USER' if m['role'] == 'user' else 'AGENT'}: {m['content'][:2000]}"
            for m in conversation
        )

        eval_prompt = f"""Evaluate whether each of these done criteria was met in the following conversation.

DONE CRITERIA:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(done_criteria))}

CONVERSATION:
{conv_text[:8000]}

Respond with ONLY a JSON object: {{"met": [1, 3, 5], "missed": [2, 4]}} where numbers are 1-based criteria indices."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": eval_prompt}],
            )
            text = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            met = [done_criteria[i - 1] for i in result.get("met", []) if 1 <= i <= len(done_criteria)]
            missed = [done_criteria[i - 1] for i in result.get("missed", []) if 1 <= i <= len(done_criteria)]
            return met, missed
        except Exception as e:
            print(f"  ⚠ Completion evaluation failed: {e}")
            return [], done_criteria


# ── Insight Validation ──────────────────────────────────────────────────────

def validate_insights(
    client,
    model: str,
    retro_content: str,
    original: SessionMetrics,
    replay: SessionMetrics,
) -> tuple[list[str], list[str]]:
    """Use AI to determine which retro insights were validated by the replay."""
    # Extract insight statements from retro
    insights = []
    for line in retro_content.split("\n"):
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            # Lines that look like insight bullets
            cleaned = line.lstrip("-* ").strip()
            if len(cleaned) > 20 and not cleaned.startswith("#"):
                insights.append(cleaned)

    # Also grab numbered playbook candidates
    in_playbook = False
    for line in retro_content.split("\n"):
        if "playbook" in line.lower() and "#" in line:
            in_playbook = True
            continue
        if in_playbook and line.strip().startswith(("1.", "2.", "3.", "4.", "5.")):
            insights.append(line.strip().lstrip("0123456789. "))
        elif in_playbook and line.startswith("#"):
            in_playbook = False

    if not insights:
        return [], []

    prompt = VALIDATE_PROMPT.format(
        retro_insights="\n".join(f"- {i}" for i in insights[:15]),
        replay_turns=replay.total_turns,
        original_turns=original.total_turns,
        replay_waste=replay.waste_ratio,
        original_waste=original.waste_ratio,
        replay_completion=replay.completion_rate,
        original_completion=original.completion_rate,
        replay_tags=json.dumps(replay.tag_distribution),
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result.get("validated", []), result.get("invalidated", [])
    except Exception as e:
        print(f"  ⚠ Insight validation failed: {e}")
        return [], []


# ── High-Level Commands ─────────────────────────────────────────────────────

def cmd_extract(session_dir: str, output: str, model: str, verbose: bool):
    """Extract optimal path from a session."""
    extractor = OptimalPathExtractor(model=model)
    prompts = extractor.extract(session_dir, verbose=verbose)

    if not prompts:
        print("  No optimal path could be extracted.")
        return

    # Save as JSONL
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p.to_dict()) + "\n")

    if verbose:
        print(f"\n  Optimal path ({len(prompts)} prompts) saved to: {output_path}")
        print(f"\n  Prompt sequence:")
        for p in prompts:
            skipped = f" (skips: {', '.join(p.skipped_ids)})" if p.skipped_ids else ""
            print(f"    {p.sequence_num}. {p.prompt_text[:80]}...{skipped}")


def cmd_auto(session_dir: str, output_dir: str, system_prompt_path: Optional[str], model: str, verbose: bool):
    """Full auto pipeline: extract → replay → compare."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Extract original metrics
    if verbose:
        print("\n📊 Extracting original session metrics...")
    original_metrics = extract_original_metrics(session_dir)
    if verbose:
        print(f"  Original: {original_metrics.total_turns} turns, {original_metrics.waste_ratio:.0%} waste")

    # 2. Extract optimal path
    if verbose:
        print("\n🔍 Extracting optimal path from retro...")
    extractor = OptimalPathExtractor(model=model)
    optimal_prompts = extractor.extract(session_dir, verbose=verbose)

    if not optimal_prompts:
        print("  ⚠ Could not extract optimal path. Aborting.")
        return

    # Save optimal path
    playbook_path = output_path / "optimal_prompts.jsonl"
    with open(playbook_path, "w") as f:
        for p in optimal_prompts:
            f.write(json.dumps(p.to_dict()) + "\n")

    # 3. Replay
    if verbose:
        print("\n🔄 Replaying optimal path...")
    system_prompt = ""
    if system_prompt_path:
        system_prompt = Path(system_prompt_path).read_text()
    else:
        # Try to find it
        candidates = [
            Path(session_dir).parent / "agent-plan-prompt-v2.md",
            Path("agent-plan-prompt-v2.md"),
        ]
        for c in candidates:
            if c.exists():
                system_prompt = c.read_text()
                break

    engine = ReplayEngine(system_prompt=system_prompt, model=model)
    replay_turns, replay_metrics = engine.replay(optimal_prompts, verbose=verbose)

    # 4. Validate insights
    retro_path = Path(session_dir) / "RETRO.md"
    validated, invalidated = [], []
    if retro_path.exists() and replay_turns:
        if verbose:
            print("\n🧪 Validating retro insights...")
        import anthropic
        client = anthropic.Anthropic()
        retro_content = retro_path.read_text()
        validated, invalidated = validate_insights(
            client, model, retro_content, original_metrics, replay_metrics
        )

    # 5. Build comparison
    comparison = ReplayComparison(
        original=original_metrics,
        replay=replay_metrics,
        optimal_path=optimal_prompts,
        insights_validated=validated,
        insights_invalidated=invalidated,
        replay_turns=replay_turns,
    )

    # Save
    (output_path / "comparison.json").write_text(json.dumps(comparison.to_dict(), indent=2))

    if verbose:
        print(comparison.summary())
        print(f"\n  Results saved to: {output_path}")


def cmd_manual(playbook_path: str, system_prompt_path: str, output_dir: str, model: str, verbose: bool):
    """Replay a manually defined optimal path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = []
    for line in Path(playbook_path).read_text().strip().split("\n"):
        if line.strip():
            try:
                data = json.loads(line)
                prompts.append(ReplayPrompt(**data))
            except (json.JSONDecodeError, TypeError):
                continue

    if not prompts:
        print("  No prompts loaded from playbook.")
        return

    system_prompt = Path(system_prompt_path).read_text() if system_prompt_path else ""

    engine = ReplayEngine(system_prompt=system_prompt, model=model)
    replay_turns, replay_metrics = engine.replay(prompts, verbose=verbose)

    # Save results
    results = {
        "metrics": replay_metrics.to_dict(),
        "turns": [t.to_dict() for t in replay_turns],
        "prompts": [p.to_dict() for p in prompts],
    }
    (output_path / "manual_replay.json").write_text(json.dumps(results, indent=2))

    if verbose:
        print(f"\n  Replay complete: {replay_metrics.total_turns} turns")
        print(f"  Productive: {replay_metrics.productive_ratio:.0%}")
        print(f"  Waste: {replay_metrics.waste_ratio:.0%}")
        print(f"  Results saved to: {output_path}")


def cmd_ab(playbook_path: str, variants: list[str], output_dir: str, system_prompt_path: Optional[str], model: str, verbose: bool):
    """A/B test: replay same path through different system prompt variants."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = []
    for line in Path(playbook_path).read_text().strip().split("\n"):
        if line.strip():
            try:
                data = json.loads(line)
                prompts.append(ReplayPrompt(**data))
            except (json.JSONDecodeError, TypeError):
                continue

    if not prompts:
        print("  No prompts loaded.")
        return

    # Define variant system prompts
    variant_prompts = {}
    for v in variants:
        if v == "full":
            if system_prompt_path:
                variant_prompts[v] = Path(system_prompt_path).read_text()
            else:
                candidates = [Path("agent-plan-prompt-v2.md"), Path.home() / "agent-plan-prompt-v2.md"]
                for c in candidates:
                    if c.exists():
                        variant_prompts[v] = c.read_text()
                        break
                else:
                    print(f"  ⚠ Can't find system prompt for 'full' variant. Use --system-prompt.")
                    return
        elif v == "minimal":
            variant_prompts[v] = MINIMAL_SYSTEM_PROMPT
        elif v == "none":
            variant_prompts[v] = ""
        else:
            print(f"  ⚠ Unknown variant: {v}")
            return

    # Run each variant
    results = {}
    for variant_name, system_prompt in variant_prompts.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Running variant: {variant_name}")
            print(f"{'='*60}")

        engine = ReplayEngine(system_prompt=system_prompt, model=model)
        replay_turns, metrics = engine.replay(prompts, verbose=verbose)
        metrics.label = variant_name

        results[variant_name] = {
            "metrics": metrics.to_dict(),
            "turns": [t.to_dict() for t in replay_turns],
        }

        # Save individual
        (output_path / f"{variant_name}_replay.json").write_text(
            json.dumps(results[variant_name], indent=2)
        )

    # Print comparison
    if verbose and len(results) >= 2:
        print(f"\n{'='*65}")
        print(f"  A/B REPLAY COMPARISON")
        print(f"{'='*65}")
        print(f"\n  {'Variant':<15} {'Turns':>8} {'Productive':>12} {'Waste':>8} {'Tokens':>10}")
        print(f"  {'-'*55}")
        for name, data in results.items():
            m = data["metrics"]
            print(f"  {name:<15} {m['total_turns']:>8} {m['productive_ratio']*100:>11.0f}% {m['waste_ratio']*100:>7.0f}% {m['total_tokens']:>10,}")

    # Save combined
    (output_path / "ab_comparison.json").write_text(json.dumps(results, indent=2))
    if verbose:
        print(f"\n  Results saved to: {output_path}")


def cmd_compare(original_dir: str, replay_dir: str, verbose: bool):
    """Compare original session vs existing replay results."""
    original_metrics = extract_original_metrics(original_dir)

    # Load replay results
    comparison_path = Path(replay_dir) / "comparison.json"
    if comparison_path.exists():
        data = json.loads(comparison_path.read_text())
        replay_metrics = SessionMetrics(**data.get("replay", {}))
    else:
        # Try manual replay
        manual_path = Path(replay_dir) / "manual_replay.json"
        if manual_path.exists():
            data = json.loads(manual_path.read_text())
            replay_metrics = SessionMetrics(**data.get("metrics", {}))
        else:
            print(f"  No replay results found in {replay_dir}")
            return

    comparison = ReplayComparison(
        original=original_metrics,
        replay=replay_metrics,
    )

    if verbose:
        print(comparison.summary())


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Session Replay Comparison — Validate retro insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full auto: extract optimal path from retro → replay → compare
  python replay.py auto --session-dir ./my-project --output replay-results/

  # Extract optimal path only (for review before replaying)
  python replay.py extract --session-dir ./my-project --output optimal_prompts.jsonl

  # Replay a hand-written optimal path
  python replay.py manual --playbook optimal.jsonl --system-prompt prompt-v2.md

  # A/B test: same path through full vs minimal system prompts
  python replay.py ab --playbook optimal.jsonl --variants full,minimal
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Auto
    auto_p = subparsers.add_parser("auto", help="Full pipeline: extract → replay → compare")
    auto_p.add_argument("--session-dir", required=True)
    auto_p.add_argument("--output", default="replay-results/")
    auto_p.add_argument("--system-prompt", help="Path to system prompt file")
    auto_p.add_argument("--model", default="claude-sonnet-4-20250514")
    auto_p.add_argument("--quiet", action="store_true")

    # Extract
    ext_p = subparsers.add_parser("extract", help="Extract optimal path from retro")
    ext_p.add_argument("--session-dir", required=True)
    ext_p.add_argument("--output", default="optimal_prompts.jsonl")
    ext_p.add_argument("--model", default="claude-sonnet-4-20250514")
    ext_p.add_argument("--quiet", action="store_true")

    # Manual
    man_p = subparsers.add_parser("manual", help="Replay a manually defined path")
    man_p.add_argument("--playbook", required=True, help="Path to JSONL of replay prompts")
    man_p.add_argument("--system-prompt", default="", help="Path to system prompt")
    man_p.add_argument("--output", default="replay-results/")
    man_p.add_argument("--model", default="claude-sonnet-4-20250514")
    man_p.add_argument("--quiet", action="store_true")

    # A/B
    ab_p = subparsers.add_parser("ab", help="A/B replay through different variants")
    ab_p.add_argument("--playbook", required=True)
    ab_p.add_argument("--variants", required=True, help="Comma-separated: full,minimal,none")
    ab_p.add_argument("--system-prompt", help="Path to 'full' system prompt file")
    ab_p.add_argument("--output", default="replay-results/")
    ab_p.add_argument("--model", default="claude-sonnet-4-20250514")
    ab_p.add_argument("--quiet", action="store_true")

    # Compare
    cmp_p = subparsers.add_parser("compare", help="Compare original vs replay results")
    cmp_p.add_argument("--original", required=True, help="Original session directory")
    cmp_p.add_argument("--replay", required=True, help="Replay results directory")
    cmp_p.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args.session_dir, args.output, args.model, not args.quiet)
    elif args.command == "auto":
        cmd_auto(args.session_dir, args.output, getattr(args, "system_prompt", None), args.model, not args.quiet)
    elif args.command == "manual":
        cmd_manual(args.playbook, args.system_prompt, args.output, args.model, not args.quiet)
    elif args.command == "ab":
        cmd_ab(args.playbook, args.variants.split(","), args.output, getattr(args, "system_prompt", None), args.model, not args.quiet)
    elif args.command == "compare":
        cmd_compare(args.original, args.replay, not args.quiet)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
