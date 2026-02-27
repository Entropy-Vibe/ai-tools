#!/usr/bin/env python3
"""
Judge Model Pipeline — Uses a separate Claude instance to evaluate agent turn quality.

Evaluates three dimensions per turn:
1. Process tag accuracy — Is the self-assigned tag correct?
2. Summary specificity — Does the summary follow naming rules?
3. Test breadcrumb completeness — Are test specs usable?

Also produces session-level judgments:
- Tag honesty score (do tags skew flattering vs. judge assessment?)
- Summary quality distribution
- Overall session quality grade

Usage:
    python judge_pipeline.py --history path/to/HISTORY.jsonl --output judgments/
    python judge_pipeline.py --history HISTORY.jsonl --test-queue TEST_QUEUE.md --output judgments/
    python judge_pipeline.py --session-dir path/to/project/ --output judgments/

Requires: ANTHROPIC_API_KEY environment variable
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Constants ───────────────────────────────────────────────────────────────

VALID_PROCESS_TAGS = {
    "clean-execution", "good-discovery", "course-correct",
    "dead-end", "redo", "yak-shave", "scope-creep", "vague-prompt", "misfire",
    "planning", "debug", "refactor",
}

PRODUCTIVE_TAGS = {"clean-execution", "good-discovery", "course-correct"}
WASTE_TAGS = {"dead-end", "redo", "yak-shave", "scope-creep", "vague-prompt", "misfire"}


# ── Judge Prompts ───────────────────────────────────────────────────────────

TAG_JUDGE_PROMPT = """You are evaluating whether a coding agent correctly self-tagged its process.

The agent uses these process tags:
PRODUCTIVE: clean-execution (clear prompt, correct response), good-discovery (learned something valuable), course-correct (caught wrong direction)
WASTE: dead-end (path abandoned), redo (should have been right first time), yak-shave (unnecessary prerequisite), scope-creep (outside the plan), vague-prompt (unclear prompt), misfire (clear prompt, wrong response)
NEUTRAL: planning (designing/scoping), debug (fixing something broken), refactor (restructuring for quality)

RULES:
- A turn that feels productive but whose work was abandoned = dead-end
- A turn that feels frustrating but redirected well = good-discovery or course-correct
- debug that recurs on same root cause: 1st = debug, 2nd = debug, 3rd = redo
- When in doubt, pick the less flattering tag

Given this prompt-response pair, evaluate the agent's self-assigned tag.

PROMPT (P-{prompt_id}):
{prompt_text}

RESPONSE (R-{response_id}):
{response_text}

AGENT'S SELF-TAG: {self_tag}

Respond with ONLY this JSON (no markdown, no explanation):
{{
  "correct_tag": "<your assessment of the correct tag>",
  "agrees": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "<one sentence explaining your judgment>"
}}"""

SUMMARY_JUDGE_PROMPT = """You are evaluating the quality of a HISTORY.jsonl summary entry from a coding agent.

Good summaries follow these rules:
- Be specific: names, files, functions, reasons. Not "updated auth."
- Include the WHY: "Switched to httpOnly cookies (XSS mitigation)"
- Name the things: `validateToken()` in `auth/middleware.js`, not "validation function"
- Capture decisions: "Chose X over Y — reason"
- Flag gotchas: "⚠️ Rate limiter is per-IP — revisit before launch"

ENTRY:
ID: {entry_id}
Type: {entry_type}
Summary: {summary}
Tags: {tags}
Milestone: {milestone}

Respond with ONLY this JSON:
{{
  "specificity": 0.0-1.0,
  "includes_why": true/false,
  "names_things": true/false,
  "captures_decisions": true/false,
  "flags_gotchas": true/false,
  "quality_grade": "excellent|good|adequate|poor|vague",
  "improvement": "<one sentence on how to improve, or 'none needed'>"
}}"""

TEST_BREADCRUMB_JUDGE_PROMPT = """You are evaluating a test breadcrumb entry from TEST_QUEUE.md.

A good breadcrumb should be a pre-chewed test spec that makes writing the actual test fast. It needs:
- What: clear description of behavior to test
- Happy path: the main success scenario
- Error states: what can go wrong
- Edge cases: boundary conditions
- Selectors: data-testid or selectors captured at build time
- Setup: auth states, fixtures, mocks needed
- Patterns: reference to existing test helpers

BREADCRUMB ENTRY:
{entry_text}

Respond with ONLY this JSON:
{{
  "completeness": 0.0-1.0,
  "has_what": true/false,
  "has_happy_path": true/false,
  "has_error_states": true/false,
  "has_edge_cases": true/false,
  "has_selectors": true/false,
  "has_setup": true/false,
  "has_patterns": true/false,
  "usability_grade": "ready_to_write|needs_minor_additions|needs_major_work|unusable",
  "missing": "<comma-separated list of what's missing, or 'nothing'>"
}}"""


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class TagJudgment:
    entry_id: str
    self_tag: str
    judge_tag: str
    agrees: bool
    confidence: float
    reasoning: str

@dataclass
class SummaryJudgment:
    entry_id: str
    specificity: float
    quality_grade: str
    includes_why: bool
    names_things: bool
    improvement: str

@dataclass
class BreadcrumbJudgment:
    entry_id: str
    completeness: float
    usability_grade: str
    missing: str

@dataclass
class SessionJudgment:
    tag_judgments: list[TagJudgment] = field(default_factory=list)
    summary_judgments: list[SummaryJudgment] = field(default_factory=list)
    breadcrumb_judgments: list[BreadcrumbJudgment] = field(default_factory=list)

    @property
    def tag_agreement_rate(self) -> float:
        if not self.tag_judgments:
            return 0.0
        return sum(1 for j in self.tag_judgments if j.agrees) / len(self.tag_judgments)

    @property
    def tag_honesty_score(self) -> float:
        """Detect flattering bias: does the agent over-tag productive vs judge assessment?"""
        if not self.tag_judgments:
            return 1.0
        agent_productive = sum(1 for j in self.tag_judgments if j.self_tag in PRODUCTIVE_TAGS)
        judge_productive = sum(1 for j in self.tag_judgments if j.judge_tag in PRODUCTIVE_TAGS)
        total = len(self.tag_judgments)
        if total == 0:
            return 1.0
        # Honesty = 1.0 if agent matches judge, lower if agent is more flattering
        agent_rate = agent_productive / total
        judge_rate = judge_productive / total
        flattering_delta = max(agent_rate - judge_rate, 0)
        return max(1.0 - (flattering_delta * 2), 0.0)  # Penalize flattering heavily

    @property
    def avg_summary_specificity(self) -> float:
        if not self.summary_judgments:
            return 0.0
        return sum(j.specificity for j in self.summary_judgments) / len(self.summary_judgments)

    @property
    def avg_breadcrumb_completeness(self) -> float:
        if not self.breadcrumb_judgments:
            return 0.0
        return sum(j.completeness for j in self.breadcrumb_judgments) / len(self.breadcrumb_judgments)

    def to_dict(self) -> dict:
        return {
            "metrics": {
                "tag_agreement_rate": round(self.tag_agreement_rate, 4),
                "tag_honesty_score": round(self.tag_honesty_score, 4),
                "avg_summary_specificity": round(self.avg_summary_specificity, 4),
                "avg_breadcrumb_completeness": round(self.avg_breadcrumb_completeness, 4),
                "tag_judgments_count": len(self.tag_judgments),
                "summary_judgments_count": len(self.summary_judgments),
                "breadcrumb_judgments_count": len(self.breadcrumb_judgments),
            },
            "tag_judgments": [asdict(j) for j in self.tag_judgments],
            "summary_judgments": [asdict(j) for j in self.summary_judgments],
            "breadcrumb_judgments": [asdict(j) for j in self.breadcrumb_judgments],
        }

    def summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("JUDGE PIPELINE REPORT")
        lines.append("=" * 60)

        lines.append(f"\n📊 Tag Accuracy")
        lines.append(f"  Agreement rate: {self.tag_agreement_rate:.1%}")
        lines.append(f"  Honesty score:  {self.tag_honesty_score:.1%} (1.0 = no flattering bias)")
        if self.tag_judgments:
            disagreements = [j for j in self.tag_judgments if not j.agrees]
            if disagreements:
                lines.append(f"  Disagreements ({len(disagreements)}):")
                for d in disagreements[:5]:
                    lines.append(f"    {d.entry_id}: agent={d.self_tag} → judge={d.judge_tag} ({d.reasoning})")

        lines.append(f"\n📝 Summary Quality")
        lines.append(f"  Avg specificity: {self.avg_summary_specificity:.1%}")
        if self.summary_judgments:
            grade_dist = {}
            for j in self.summary_judgments:
                grade_dist[j.quality_grade] = grade_dist.get(j.quality_grade, 0) + 1
            lines.append(f"  Grade distribution: {grade_dist}")
            poor = [j for j in self.summary_judgments if j.quality_grade in ("poor", "vague")]
            if poor:
                lines.append(f"  Needs improvement ({len(poor)}):")
                for p in poor[:5]:
                    lines.append(f"    {p.entry_id}: {p.improvement}")

        lines.append(f"\n🧪 Test Breadcrumb Quality")
        lines.append(f"  Avg completeness: {self.avg_breadcrumb_completeness:.1%}")
        if self.breadcrumb_judgments:
            usability_dist = {}
            for j in self.breadcrumb_judgments:
                usability_dist[j.usability_grade] = usability_dist.get(j.usability_grade, 0) + 1
            lines.append(f"  Usability distribution: {usability_dist}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ── Judge Engine ────────────────────────────────────────────────────────────

class JudgePipeline:
    """Runs judge evaluations using a separate Claude instance."""

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

    def _call_judge(self, prompt: str) -> dict:
        """Make a judge API call and parse JSON response."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"  ⚠ Judge returned non-JSON: {text[:200]}")
            return {}
        except Exception as e:
            print(f"  ⚠ Judge call failed: {e}")
            return {}

    def judge_tags(self, entries: list[dict], verbose: bool = True) -> list[TagJudgment]:
        """Judge process tag accuracy for response entries."""
        judgments = []
        response_entries = [e for e in entries if e.get("type") == "response" and e.get("process_tag")]

        # Pair responses with their prompts
        prompt_map = {e["id"]: e for e in entries if e.get("type") == "prompt"}

        for i, entry in enumerate(response_entries):
            response_id = entry["id"]
            prompt_id = response_id.replace("R-", "P-")
            prompt_entry = prompt_map.get(prompt_id, {})

            if verbose:
                print(f"  Judging tag for {response_id} ({i+1}/{len(response_entries)})...")

            prompt = TAG_JUDGE_PROMPT.format(
                prompt_id=prompt_id.replace("P-", ""),
                response_id=response_id.replace("R-", ""),
                prompt_text=prompt_entry.get("content", prompt_entry.get("summary", "[no content]"))[:1500],
                response_text=entry.get("content", entry.get("summary", "[no content]"))[:1500],
                self_tag=entry["process_tag"],
            )

            result = self._call_judge(prompt)
            if result:
                judgments.append(TagJudgment(
                    entry_id=response_id,
                    self_tag=entry["process_tag"],
                    judge_tag=result.get("correct_tag", "unknown"),
                    agrees=result.get("agrees", False),
                    confidence=result.get("confidence", 0.0),
                    reasoning=result.get("reasoning", ""),
                ))

            # Rate limiting
            time.sleep(0.5)

        return judgments

    def judge_summaries(self, entries: list[dict], verbose: bool = True) -> list[SummaryJudgment]:
        """Judge summary quality for all entries."""
        judgments = []

        for i, entry in enumerate(entries):
            if not entry.get("summary"):
                continue

            if verbose:
                print(f"  Judging summary for {entry['id']} ({i+1}/{len(entries)})...")

            prompt = SUMMARY_JUDGE_PROMPT.format(
                entry_id=entry["id"],
                entry_type=entry.get("type", "unknown"),
                summary=entry["summary"],
                tags=json.dumps(entry.get("tags", [])),
                milestone=entry.get("milestone", "unknown"),
            )

            result = self._call_judge(prompt)
            if result:
                judgments.append(SummaryJudgment(
                    entry_id=entry["id"],
                    specificity=result.get("specificity", 0.0),
                    quality_grade=result.get("quality_grade", "unknown"),
                    includes_why=result.get("includes_why", False),
                    names_things=result.get("names_things", False),
                    improvement=result.get("improvement", ""),
                ))

            time.sleep(0.3)

        return judgments

    def judge_breadcrumbs(self, tq_text: str, verbose: bool = True) -> list[BreadcrumbJudgment]:
        """Judge test breadcrumb quality from TEST_QUEUE.md."""
        import re
        judgments = []

        # Split into individual entries
        entries = re.split(r"(?=^### TQ-\d+:)", tq_text, flags=re.MULTILINE)
        entries = [e.strip() for e in entries if e.strip().startswith("### TQ-")]

        for i, entry_text in enumerate(entries):
            entry_id_match = re.match(r"### (TQ-\d+)", entry_text)
            entry_id = entry_id_match.group(1) if entry_id_match else f"TQ-{i}"

            if verbose:
                print(f"  Judging breadcrumb {entry_id} ({i+1}/{len(entries)})...")

            prompt = TEST_BREADCRUMB_JUDGE_PROMPT.format(entry_text=entry_text[:2000])

            result = self._call_judge(prompt)
            if result:
                judgments.append(BreadcrumbJudgment(
                    entry_id=entry_id,
                    completeness=result.get("completeness", 0.0),
                    usability_grade=result.get("usability_grade", "unknown"),
                    missing=result.get("missing", ""),
                ))

            time.sleep(0.3)

        return judgments

    def run(
        self,
        history_path: Optional[str] = None,
        test_queue_path: Optional[str] = None,
        verbose: bool = True,
    ) -> SessionJudgment:
        """Run the full judge pipeline."""
        judgment = SessionJudgment()

        if history_path:
            if verbose:
                print("\n📊 Judging process tags...")
            history_text = Path(history_path).read_text()
            entries = []
            for line in history_text.strip().split("\n"):
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            judgment.tag_judgments = self.judge_tags(entries, verbose)

            if verbose:
                print("\n📝 Judging summary quality...")
            judgment.summary_judgments = self.judge_summaries(entries, verbose)

        if test_queue_path:
            if verbose:
                print("\n🧪 Judging test breadcrumbs...")
            tq_text = Path(test_queue_path).read_text()
            judgment.breadcrumb_judgments = self.judge_breadcrumbs(tq_text, verbose)

        return judgment


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Judge Pipeline for Agent Plan Prompt v2")
    parser.add_argument("--history", help="Path to HISTORY.jsonl")
    parser.add_argument("--test-queue", help="Path to TEST_QUEUE.md")
    parser.add_argument("--session-dir", help="Path to project directory (auto-discovers files)")
    parser.add_argument("--output", default="judgments/", help="Output directory")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--json", action="store_true", help="Output as JSON only")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Auto-discover from session directory
    if args.session_dir:
        session_dir = Path(args.session_dir)
        if not args.history and (session_dir / "HISTORY.jsonl").exists():
            args.history = str(session_dir / "HISTORY.jsonl")
        if not args.test_queue and (session_dir / "TEST_QUEUE.md").exists():
            args.test_queue = str(session_dir / "TEST_QUEUE.md")

    if not args.history and not args.test_queue:
        parser.error("Provide --history, --test-queue, or --session-dir")

    pipeline = JudgePipeline(model=args.model)
    judgment = pipeline.run(
        history_path=args.history,
        test_queue_path=args.test_queue,
        verbose=not args.quiet,
    )

    # Output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"judgment_{timestamp}.json"
    output_path.write_text(json.dumps(judgment.to_dict(), indent=2))

    if args.json:
        print(json.dumps(judgment.to_dict(), indent=2))
    else:
        print(judgment.summary())
        print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
