#!/usr/bin/env python3
"""
Prompt Quality Analyzer — Evaluates YOUR prompting patterns.
=============================================================

Unlike the Judge Pipeline (which evaluates the agent's output), this tool
evaluates the HUMAN's prompts for patterns that lead to waste turns.

Analyzes three layers:
1. Per-prompt quality — Is this prompt specific enough? Does it follow playbook patterns?
2. Sequence patterns — Are you falling into scope-creep loops, vague→correct cycles, detail spirals?
3. Session habits — Over-time trends in your prompting behavior.

Can run in two modes:
- Post-hoc analysis: Feed it a HISTORY.jsonl to analyze a completed session
- Live check: Feed it a single prompt to score before sending to the agent

Usage:
    # Analyze a full session
    python prompt_analyzer.py session --history path/to/HISTORY.jsonl
    python prompt_analyzer.py session --session-dir path/to/project/

    # Score a single prompt before sending
    python prompt_analyzer.py check "Build the auth system with JWT tokens"
    python prompt_analyzer.py check "Make it work" --context "Building a REST API"

    # Analyze prompting habits across multiple sessions
    python prompt_analyzer.py habits --dirs session1/ session2/ session3/

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


# ── Anti-Patterns & Playbook Rules ──────────────────────────────────────────
# These are derived from the system prompt's PLAYBOOK.md patterns

PROMPT_ANTIPATTERNS = {
    "vague_verb": {
        "patterns": [
            r"^(make|do|fix|handle|update|change|add|set up|build)\s+(it|this|that|the\s+\w+)\s*$",
            r"^(make|do|fix|handle)\s+\w+\s+(work|better|right|properly)\s*$",
            r"^(work on|deal with|take care of|look at)\s+",
        ],
        "description": "Vague verb + unspecified target",
        "severity": "high",
        "fix": "Specify: input → operation → output. Name files, functions, expected behavior.",
        "example_bad": "Make the auth work",
        "example_good": "Implement JWT middleware in auth/middleware.js: validate token from Authorization header, check expiry, return 401 with {error: 'token_expired'}",
    },
    "kitchen_sink": {
        "patterns": [
            r"(?:also|and also|plus|oh and|while you're at it|and then|additionally)",
        ],
        "description": "Multiple unrelated concerns in one prompt",
        "severity": "medium",
        "fix": "One concern per prompt. Sequence: scaffold → happy path → errors → edge cases.",
        "example_bad": "Build the login form and also fix the header alignment and add a loading spinner",
        "example_good": "Build the login form with email/password fields, submit button, and validation messages",
    },
    "no_output_spec": {
        "patterns": [
            # Detected heuristically, not via regex — checked in analysis
        ],
        "description": "No specification of expected output shape, format, or behavior",
        "severity": "medium",
        "fix": "Define what 'done' looks like. What does the function return? What status code? What error format?",
    },
    "premature_optimization": {
        "patterns": [
            r"(blazing fast|high performance|scale to millions|optimize|efficient|cache everything|maximum throughput)",
        ],
        "description": "Performance requirements before functionality exists",
        "severity": "medium",
        "fix": "Build simple first, optimize only after proving it's a bottleneck. Measure before caching.",
    },
    "assumed_context": {
        "patterns": [
            r"^(continue|keep going|next|do the next|more|same thing|like before)\s*\.?$",
        ],
        "description": "Assumes the agent remembers implicit context",
        "severity": "low",
        "fix": "Reference specific: 'Continue from step 3 of the auth middleware — implement refresh token rotation.'",
    },
    "mega_prompt": {
        "patterns": [
            # Detected by length + density, not regex
        ],
        "description": "Single prompt trying to specify an entire system at once",
        "severity": "high",
        "fix": "Decompose: scaffold → data model → core logic → error handling → integration. One prompt per layer.",
    },
    "detail_spiral": {
        "patterns": [
            r"(actually|wait|no,|hmm|on second thought|oh also|I forgot|one more thing)",
        ],
        "description": "Stream-of-consciousness corrections mid-prompt",
        "severity": "low",
        "fix": "Pause, organize your requirements, then send one clean prompt instead of thinking out loud.",
    },
    "implicit_quality": {
        "patterns": [
            r"(make it good|make it nice|make it clean|do it properly|do it right|best practices|production.?ready)",
        ],
        "description": "Quality requirements without concrete criteria",
        "severity": "medium",
        "fix": "Define 'good' concretely: error handling, input validation, logging, test coverage, specific patterns.",
    },
}

# Sequence patterns — detected across consecutive prompts
SEQUENCE_ANTIPATTERNS = {
    "vague_then_correct": {
        "description": "Vague prompt → agent guesses wrong → you correct with specifics you should have included",
        "waste_cost": "1-2 wasted turns per occurrence",
        "fix": "Include the corrective details in the first prompt.",
    },
    "scope_creep_spiral": {
        "description": "Planned task → notice unrelated issue → address it → notice another → lose original thread",
        "waste_cost": "2-5 wasted turns per spiral",
        "fix": "Note the issue in Active Context, finish current task first, address later.",
    },
    "debug_loop": {
        "description": "Same bug addressed 3+ times because root cause wasn't identified",
        "waste_cost": "2+ redo turns",
        "fix": "After second debug attempt, stop and do root cause analysis before more fixes.",
    },
    "premature_detail": {
        "description": "Specifying implementation details before the high-level approach is agreed on",
        "waste_cost": "Redos when approach changes",
        "fix": "Agree on approach first (planning turn), then specify details.",
    },
    "ping_pong": {
        "description": "Alternating between building and questioning the approach, never committing",
        "waste_cost": "Stalled progress",
        "fix": "Commit to an approach for at least 3-5 turns before re-evaluating.",
    },
}


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class PromptScore:
    """Score for a single prompt."""
    prompt_id: str
    prompt_text: str
    overall_score: float  # 0.0-1.0
    specificity: float
    actionability: float
    scope_clarity: float
    output_defined: bool
    antipatterns_triggered: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    rewrite: str = ""  # Suggested rewrite
    grade: str = ""  # A/B/C/D/F

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SequencePattern:
    """Detected sequence-level antipattern."""
    pattern_name: str
    description: str
    prompt_range: str  # e.g. "P-005 → P-008"
    evidence: str
    fix: str
    waste_turns: int


@dataclass
class SessionAnalysis:
    """Full session analysis of prompt quality."""
    prompt_scores: list[PromptScore] = field(default_factory=list)
    sequence_patterns: list[SequencePattern] = field(default_factory=list)
    habit_summary: dict = field(default_factory=dict)

    @property
    def avg_score(self) -> float:
        if not self.prompt_scores:
            return 0.0
        return sum(p.overall_score for p in self.prompt_scores) / len(self.prompt_scores)

    @property
    def grade_distribution(self) -> dict:
        dist = {}
        for p in self.prompt_scores:
            dist[p.grade] = dist.get(p.grade, 0) + 1
        return dist

    @property
    def most_common_antipatterns(self) -> list[tuple[str, int]]:
        counts = {}
        for p in self.prompt_scores:
            for ap in p.antipatterns_triggered:
                counts[ap] = counts.get(ap, 0) + 1
        return sorted(counts.items(), key=lambda x: -x[1])

    @property
    def estimated_waste_from_prompts(self) -> int:
        return sum(sp.waste_turns for sp in self.sequence_patterns)

    def to_dict(self) -> dict:
        return {
            "metrics": {
                "avg_prompt_score": round(self.avg_score, 4),
                "grade_distribution": self.grade_distribution,
                "most_common_antipatterns": self.most_common_antipatterns,
                "sequence_patterns_found": len(self.sequence_patterns),
                "estimated_waste_from_prompts": self.estimated_waste_from_prompts,
                "total_prompts_analyzed": len(self.prompt_scores),
            },
            "prompt_scores": [p.to_dict() for p in self.prompt_scores],
            "sequence_patterns": [asdict(sp) for sp in self.sequence_patterns],
            "habit_summary": self.habit_summary,
        }

    def summary(self) -> str:
        lines = []
        lines.append("=" * 65)
        lines.append("  PROMPT QUALITY ANALYSIS")
        lines.append("=" * 65)

        # Overall
        avg = self.avg_score * 100
        bar = _score_bar(self.avg_score)
        lines.append(f"\n  Overall Prompt Score: {avg:.0f}/100  {bar}")
        lines.append(f"  Prompts analyzed: {len(self.prompt_scores)}")

        # Grade distribution
        if self.grade_distribution:
            grades = " · ".join(f"{g}: {c}" for g, c in sorted(self.grade_distribution.items()))
            lines.append(f"  Grades: {grades}")

        # Top antipatterns
        if self.most_common_antipatterns:
            lines.append(f"\n  ⚠  Most Common Antipatterns:")
            for name, count in self.most_common_antipatterns[:5]:
                ap = PROMPT_ANTIPATTERNS.get(name, {})
                desc = ap.get("description", name)
                lines.append(f"     {count}× {name}: {desc}")

        # Sequence patterns
        if self.sequence_patterns:
            lines.append(f"\n  🔄 Sequence Patterns Detected ({len(self.sequence_patterns)}):")
            for sp in self.sequence_patterns:
                lines.append(f"     {sp.prompt_range}: {sp.pattern_name} (~{sp.waste_turns} waste turns)")
                lines.append(f"       → {sp.fix}")

        # Estimated waste
        if self.estimated_waste_from_prompts > 0:
            lines.append(f"\n  📉 Estimated turns wasted due to prompt quality: {self.estimated_waste_from_prompts}")
            total = len(self.prompt_scores)
            if total > 0:
                lines.append(f"     That's ~{self.estimated_waste_from_prompts / total:.0%} of your session spent on avoidable rework")

        # Worst prompts
        worst = sorted(self.prompt_scores, key=lambda p: p.overall_score)[:3]
        if worst and worst[0].overall_score < 0.7:
            lines.append(f"\n  📋 Prompts Most Needing Improvement:")
            for p in worst:
                if p.overall_score < 0.7:
                    lines.append(f"     {p.prompt_id} (score: {p.overall_score:.0%}): {p.prompt_text[:80]}...")
                    if p.rewrite:
                        lines.append(f"       ✏️  Better: {p.rewrite[:100]}...")

        # Habit summary
        if self.habit_summary:
            lines.append(f"\n  🧠 Habit Profile:")
            for habit, info in self.habit_summary.items():
                lines.append(f"     {info}")

        lines.append("\n" + "=" * 65)
        return "\n".join(lines)


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    empty = width - filled
    if score >= 0.8:
        color = "\033[92m"  # green
    elif score >= 0.6:
        color = "\033[93m"  # yellow
    else:
        color = "\033[91m"  # red
    return f"{color}{'█' * filled}{'░' * empty}\033[0m"


def _score_to_grade(score: float) -> str:
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.65:
        return "C"
    elif score >= 0.5:
        return "D"
    return "F"


# ── Local (Regex-based) Analyzers ───────────────────────────────────────────
# These run without an API call — fast, free, catches obvious issues

def analyze_prompt_local(prompt_id: str, prompt_text: str) -> PromptScore:
    """Fast, regex-based prompt analysis. No API call needed."""
    text = prompt_text.strip()
    lower = text.lower()
    triggered = []
    suggestions = []

    # ── Antipattern detection ──
    for name, ap in PROMPT_ANTIPATTERNS.items():
        for pattern in ap.get("patterns", []):
            if re.search(pattern, lower):
                triggered.append(name)
                suggestions.append(ap["fix"])
                break

    # ── Length-based checks ──
    word_count = len(text.split())

    # Mega-prompt detection
    if word_count > 300:
        triggered.append("mega_prompt")
        suggestions.append(PROMPT_ANTIPATTERNS["mega_prompt"]["fix"])
    # Too-short prompt detection (likely vague)
    if word_count < 5 and not any(t in triggered for t in ["assumed_context", "vague_verb"]):
        triggered.append("assumed_context")
        suggestions.append("Prompts under 5 words rarely contain enough specification.")

    # ── Output specification check ──
    output_keywords = [
        "return", "returns", "respond", "output", "produces", "yields",
        "status code", "→", "->", "should return", "should output",
        "expected", "format:", "shape:", "schema:",
    ]
    output_defined = any(kw in lower for kw in output_keywords)
    if not output_defined and word_count > 15:
        # Only flag this for substantial prompts, not brief continuations
        triggered.append("no_output_spec")
        suggestions.append(PROMPT_ANTIPATTERNS["no_output_spec"]["fix"])

    # ── Specificity scoring ──
    # Indicators of specificity
    specificity_signals = [
        bool(re.search(r'`[^`]+`', text)),           # Code references
        bool(re.search(r'\w+\.\w+', text)),           # file.ext references
        bool(re.search(r'\w+\(\)', text)),             # function() references
        bool(re.search(r'(POST|GET|PUT|DELETE|PATCH)\s+/', text)),  # HTTP methods
        bool(re.search(r'\{[^}]+\}', text)),           # JSON/object shapes
        bool(re.search(r'\d{3}', text)),               # Status codes
        bool(re.search(r'(if|when|unless|error|fail)', lower)),  # Condition specification
        word_count >= 20,                              # Reasonable length
    ]
    specificity = sum(specificity_signals) / len(specificity_signals)

    # ── Actionability scoring ──
    action_signals = [
        bool(re.search(r'^(implement|create|build|add|write|configure|set up|design|refactor|extract|move|rename)', lower)),
        word_count >= 10,
        not bool(re.search(r'^(what|how|why|can you|could you|would you)', lower)),  # Not a question
        "should" in lower or "must" in lower or "need" in lower,
    ]
    actionability = sum(action_signals) / len(action_signals)

    # ── Scope clarity ──
    scope_signals = [
        len(triggered) == 0 or "kitchen_sink" not in triggered,  # Not multi-concern
        len(triggered) == 0 or "mega_prompt" not in triggered,   # Not a wall of text
        "only" in lower or "just" in lower or "specifically" in lower,  # Explicit scope bounds
        word_count < 200,  # Not overwhelming
    ]
    scope_clarity = sum(scope_signals) / len(scope_signals)

    # ── Overall score ──
    # Weight: specificity matters most, then actionability, then scope
    severity_penalty = sum(
        {"high": 0.15, "medium": 0.08, "low": 0.03}.get(
            PROMPT_ANTIPATTERNS.get(t, {}).get("severity", "low"), 0.03
        )
        for t in triggered
    )
    raw_score = (specificity * 0.4 + actionability * 0.3 + scope_clarity * 0.3)
    overall = max(0.0, min(1.0, raw_score - severity_penalty))

    # Deduplicate suggestions
    suggestions = list(dict.fromkeys(suggestions))

    return PromptScore(
        prompt_id=prompt_id,
        prompt_text=text[:500],
        overall_score=round(overall, 4),
        specificity=round(specificity, 4),
        actionability=round(actionability, 4),
        scope_clarity=round(scope_clarity, 4),
        output_defined=output_defined,
        antipatterns_triggered=triggered,
        suggestions=suggestions,
        grade=_score_to_grade(overall),
    )


# ── Sequence Analysis ───────────────────────────────────────────────────────

def analyze_sequences(entries: list[dict]) -> list[SequencePattern]:
    """Detect multi-turn antipatterns in prompt/response sequences."""
    patterns_found = []

    prompts = [e for e in entries if e.get("type") == "prompt"]
    responses = [e for e in entries if e.get("type") == "response"]
    response_map = {}
    for r in responses:
        num = r["id"].replace("R-", "")
        response_map[num] = r

    # ── Vague → Correct cycles ──
    for i in range(len(prompts) - 1):
        p1 = prompts[i]
        r1 = response_map.get(p1["id"].replace("P-", ""), {})
        p2 = prompts[i + 1] if i + 1 < len(prompts) else None

        if not p2:
            continue

        # If response was tagged vague-prompt or misfire, and next prompt is more specific
        r1_tag = r1.get("process_tag", "")
        if r1_tag in ("vague-prompt", "misfire"):
            p2_words = len(p2.get("content", p2.get("summary", "")).split())
            p1_words = len(p1.get("content", p1.get("summary", "")).split())
            if p2_words > p1_words * 1.5:  # Next prompt significantly more detailed
                patterns_found.append(SequencePattern(
                    pattern_name="vague_then_correct",
                    description=SEQUENCE_ANTIPATTERNS["vague_then_correct"]["description"],
                    prompt_range=f"{p1['id']} → {p2['id']}",
                    evidence=f"'{p1.get('summary', '')[:60]}' tagged {r1_tag}, followed by longer correction",
                    fix=SEQUENCE_ANTIPATTERNS["vague_then_correct"]["fix"],
                    waste_turns=1,
                ))

    # ── Scope creep spirals ──
    consecutive_creep = 0
    creep_start = None
    for r in responses:
        tag = r.get("process_tag", "")
        if tag in ("scope-creep", "yak-shave"):
            if consecutive_creep == 0:
                creep_start = r["id"]
            consecutive_creep += 1
        else:
            if consecutive_creep >= 2:
                patterns_found.append(SequencePattern(
                    pattern_name="scope_creep_spiral",
                    description=SEQUENCE_ANTIPATTERNS["scope_creep_spiral"]["description"],
                    prompt_range=f"{creep_start} → {r['id']}",
                    evidence=f"{consecutive_creep} consecutive scope-creep/yak-shave turns",
                    fix=SEQUENCE_ANTIPATTERNS["scope_creep_spiral"]["fix"],
                    waste_turns=consecutive_creep,
                ))
            consecutive_creep = 0
            creep_start = None

    # ── Debug loops ──
    debug_count = 0
    debug_start = None
    debug_summaries = []
    for r in responses:
        tag = r.get("process_tag", "")
        if tag == "debug":
            if debug_count == 0:
                debug_start = r["id"]
            debug_count += 1
            debug_summaries.append(r.get("summary", ""))
        else:
            if debug_count >= 3:
                patterns_found.append(SequencePattern(
                    pattern_name="debug_loop",
                    description=SEQUENCE_ANTIPATTERNS["debug_loop"]["description"],
                    prompt_range=f"{debug_start} → {r['id']}",
                    evidence=f"{debug_count} debug turns: {'; '.join(s[:40] for s in debug_summaries[:3])}",
                    fix=SEQUENCE_ANTIPATTERNS["debug_loop"]["fix"],
                    waste_turns=max(0, debug_count - 2),  # First 2 debug turns are normal
                ))
            debug_count = 0
            debug_start = None
            debug_summaries = []

    # ── Ping-pong (alternating planning/building) ──
    last_was_planning = False
    switches = 0
    switch_start = None
    for r in responses:
        tag = r.get("process_tag", "")
        is_planning = tag == "planning"
        if is_planning != last_was_planning:
            switches += 1
            if switches == 1:
                switch_start = r["id"]
        else:
            if switches >= 4:  # 4+ plan/build alternations
                patterns_found.append(SequencePattern(
                    pattern_name="ping_pong",
                    description=SEQUENCE_ANTIPATTERNS["ping_pong"]["description"],
                    prompt_range=f"{switch_start} → {r['id']}",
                    evidence=f"{switches} planning/building alternations",
                    fix=SEQUENCE_ANTIPATTERNS["ping_pong"]["fix"],
                    waste_turns=switches // 2,
                ))
            switches = 0
        last_was_planning = is_planning

    return patterns_found


# ── AI-Enhanced Analysis ────────────────────────────────────────────────────

PROMPT_JUDGE_PROMPT = """You are evaluating the quality of a HUMAN's prompt to a coding agent.

A good prompt to a coding agent:
- Specifies input, operation, and output explicitly
- Names files, functions, types, and expected behavior
- Defines what 'done' looks like
- Stays focused on one concern per prompt
- Sequences work appropriately (data model → CRUD → happy path → errors → edge cases)

A bad prompt:
- Is vague ("make it work", "fix the auth")
- Combines multiple unrelated tasks
- Specifies implementation details without agreeing on approach first
- Asks for optimization before functionality exists
- Contains stream-of-consciousness corrections

PROMPT:
{prompt_text}

CONTEXT (if available):
Prior prompt: {prior_prompt}
Agent's response to this prompt was tagged: {response_tag}

Respond with ONLY this JSON:
{{
  "overall_score": 0.0-1.0,
  "specificity": 0.0-1.0,
  "actionability": 0.0-1.0,
  "scope_clarity": 0.0-1.0,
  "output_defined": true/false,
  "antipatterns": ["list", "of", "triggered", "patterns"],
  "grade": "A|B|C|D|F",
  "rewrite": "<improved version of this prompt in 1-3 sentences, or 'none needed'>",
  "reasoning": "<one sentence on the biggest issue>"
}}"""


class PromptAnalyzer:
    """Analyzes prompt quality using both local rules and AI judge."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None, use_ai: bool = True):
        self.model = model
        self.use_ai = use_ai
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None

        if use_ai:
            if not self.api_key:
                print("⚠ No ANTHROPIC_API_KEY found. Running in local-only mode.")
                self.use_ai = False
            else:
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=self.api_key)
                except ImportError:
                    print("⚠ anthropic not installed. Running in local-only mode.")
                    self.use_ai = False

    def _call_ai(self, prompt: str) -> dict:
        if not self.client:
            return {}
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
        except Exception as e:
            print(f"  ⚠ AI call failed: {e}")
            return {}

    def score_prompt(
        self,
        prompt_id: str,
        prompt_text: str,
        prior_prompt: str = "",
        response_tag: str = "",
        verbose: bool = False,
    ) -> PromptScore:
        """Score a single prompt using local rules + optional AI enhancement."""
        # Always start with local analysis
        score = analyze_prompt_local(prompt_id, prompt_text)

        # Enhance with AI if available
        if self.use_ai:
            if verbose:
                print(f"  🤖 AI-scoring {prompt_id}...")

            ai_prompt = PROMPT_JUDGE_PROMPT.format(
                prompt_text=prompt_text[:2000],
                prior_prompt=prior_prompt[:500] if prior_prompt else "[first prompt in session]",
                response_tag=response_tag or "[not available]",
            )

            result = self._call_ai(ai_prompt)
            if result:
                # Merge AI scores with local (AI takes precedence where available)
                score.overall_score = round(result.get("overall_score", score.overall_score), 4)
                score.specificity = round(result.get("specificity", score.specificity), 4)
                score.actionability = round(result.get("actionability", score.actionability), 4)
                score.scope_clarity = round(result.get("scope_clarity", score.scope_clarity), 4)
                score.output_defined = result.get("output_defined", score.output_defined)
                score.grade = result.get("grade", score.grade)
                score.rewrite = result.get("rewrite", "")

                # Merge antipatterns (union of local + AI detected)
                ai_patterns = result.get("antipatterns", [])
                all_patterns = list(dict.fromkeys(score.antipatterns_triggered + ai_patterns))
                score.antipatterns_triggered = all_patterns

            time.sleep(0.3)

        return score

    def analyze_session(
        self,
        history_path: str,
        verbose: bool = True,
    ) -> SessionAnalysis:
        """Analyze all prompts in a HISTORY.jsonl session."""
        analysis = SessionAnalysis()

        # Load entries
        entries = []
        for line in Path(history_path).read_text().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        prompts = [e for e in entries if e.get("type") == "prompt"]
        response_map = {e["id"].replace("R-", ""): e for e in entries if e.get("type") == "response"}

        if verbose:
            print(f"\n📊 Analyzing {len(prompts)} prompts...")

        # Score each prompt
        prior_text = ""
        for i, prompt_entry in enumerate(prompts):
            pid = prompt_entry["id"]
            text = prompt_entry.get("content", prompt_entry.get("summary", ""))
            rid = pid.replace("P-", "")
            response_tag = response_map.get(rid, {}).get("process_tag", "")

            if verbose:
                print(f"  [{i+1}/{len(prompts)}] {pid}: {text[:60]}...")

            score = self.score_prompt(
                prompt_id=pid,
                prompt_text=text,
                prior_prompt=prior_text,
                response_tag=response_tag,
                verbose=verbose,
            )
            analysis.prompt_scores.append(score)
            prior_text = text

        # Sequence analysis
        if verbose:
            print(f"\n🔄 Analyzing prompt sequences...")
        analysis.sequence_patterns = analyze_sequences(entries)

        # Build habit summary
        analysis.habit_summary = self._build_habit_summary(analysis)

        return analysis

    def _build_habit_summary(self, analysis: SessionAnalysis) -> dict:
        """Build a human-readable habit profile."""
        habits = {}
        scores = analysis.prompt_scores
        if not scores:
            return habits

        # Specificity habit
        avg_spec = sum(p.specificity for p in scores) / len(scores)
        if avg_spec >= 0.7:
            habits["specificity"] = "✅ Specificity: Strong — you name files, functions, and expected behavior consistently"
        elif avg_spec >= 0.5:
            habits["specificity"] = "⚠️ Specificity: Mixed — some prompts are specific, others rely on context the agent may not have"
        else:
            habits["specificity"] = "❌ Specificity: Weak — most prompts lack concrete details. Try: name the file, function, and expected output every time."

        # Scope control
        creep_count = sum(1 for p in scores if "kitchen_sink" in p.antipatterns_triggered or "mega_prompt" in p.antipatterns_triggered)
        creep_ratio = creep_count / len(scores) if scores else 0
        if creep_ratio < 0.1:
            habits["scope"] = "✅ Scope control: Strong — you stay focused on one concern per prompt"
        elif creep_ratio < 0.25:
            habits["scope"] = "⚠️ Scope control: Drifting — about 1 in 5 prompts tries to do too much at once"
        else:
            habits["scope"] = "❌ Scope control: Weak — you frequently combine multiple concerns. Sequence them instead."

        # Output specification
        output_count = sum(1 for p in scores if p.output_defined)
        output_ratio = output_count / len(scores) if scores else 0
        if output_ratio >= 0.6:
            habits["output_spec"] = "✅ Output specification: Good — you usually define what 'done' looks like"
        elif output_ratio >= 0.3:
            habits["output_spec"] = "⚠️ Output specification: Inconsistent — define expected return values, status codes, and error shapes more often"
        else:
            habits["output_spec"] = "❌ Output specification: Rare — the agent is guessing what you want. Add expected outputs to every build prompt."

        # Detail spiral tendency
        spiral_count = sum(1 for p in scores if "detail_spiral" in p.antipatterns_triggered)
        if spiral_count >= 3:
            habits["detail_spiral"] = f"⚠️ Detail spiral: Detected in {spiral_count} prompts — you tend to think out loud and add corrections mid-stream. Draft prompts before sending."
        
        # Vague start tendency
        vague_count = sum(1 for p in scores if "vague_verb" in p.antipatterns_triggered or "assumed_context" in p.antipatterns_triggered)
        if vague_count >= 2:
            habits["vague_starts"] = f"⚠️ Vague starts: {vague_count} prompts opened vaguely. Your correction turns after these are the specifics you should lead with."

        # Sequence pattern habits
        seq_names = [sp.pattern_name for sp in analysis.sequence_patterns]
        if seq_names.count("vague_then_correct") >= 2:
            habits["vague_correct_cycle"] = "🔄 Repeat pattern: vague prompt → wrong result → specific correction. Break the cycle by front-loading details."

        return habits

    def check_single(self, prompt_text: str, context: str = "", verbose: bool = True) -> PromptScore:
        """Quick-check a single prompt before sending it to the agent."""
        score = self.score_prompt(
            prompt_id="LIVE",
            prompt_text=prompt_text,
            prior_prompt=context,
            verbose=verbose,
        )

        if not verbose:
            return score

        # Pretty-print for live use
        bar = _score_bar(score.overall_score)
        print(f"\n  Score: {score.overall_score * 100:.0f}/100 {bar}  Grade: {score.grade}")

        if score.antipatterns_triggered:
            print(f"  ⚠ Antipatterns: {', '.join(score.antipatterns_triggered)}")
            for suggestion in score.suggestions[:3]:
                print(f"    → {suggestion}")

        if score.rewrite and score.rewrite.lower() != "none needed":
            print(f"\n  ✏️  Suggested rewrite:")
            print(f"     {score.rewrite}")

        if not score.antipatterns_triggered and score.overall_score >= 0.8:
            print(f"  ✅ This prompt looks good — specific, scoped, and actionable.")

        return score


# ── Cross-Session Habits ────────────────────────────────────────────────────

def analyze_habits(session_dirs: list[str], verbose: bool = True) -> dict:
    """Analyze prompting habits across multiple sessions."""
    analyzer = PromptAnalyzer(use_ai=False)  # Local-only for speed across many sessions
    all_scores = []
    all_sequences = []
    session_summaries = []

    for d in session_dirs:
        history = Path(d) / "HISTORY.jsonl"
        if not history.exists():
            continue

        if verbose:
            print(f"\n  Analyzing {d}...")

        analysis = analyzer.analyze_session(str(history), verbose=False)
        all_scores.extend(analysis.prompt_scores)
        all_sequences.extend(analysis.sequence_patterns)
        session_summaries.append({
            "dir": d,
            "prompts": len(analysis.prompt_scores),
            "avg_score": analysis.avg_score,
            "waste_from_prompts": analysis.estimated_waste_from_prompts,
        })

    if not all_scores:
        return {"error": "No prompts found in provided sessions"}

    # Aggregate
    total_prompts = len(all_scores)
    avg_score = sum(p.overall_score for p in all_scores) / total_prompts
    total_waste = sum(sp.waste_turns for sp in all_sequences)

    # Antipattern frequency across all sessions
    ap_counts = {}
    for p in all_scores:
        for ap in p.antipatterns_triggered:
            ap_counts[ap] = ap_counts.get(ap, 0) + 1
    top_antipatterns = sorted(ap_counts.items(), key=lambda x: -x[1])

    # Trend: are scores improving across sessions?
    if len(session_summaries) >= 3:
        first_half = session_summaries[:len(session_summaries) // 2]
        second_half = session_summaries[len(session_summaries) // 2:]
        first_avg = sum(s["avg_score"] for s in first_half) / len(first_half)
        second_avg = sum(s["avg_score"] for s in second_half) / len(second_half)
        trend = "improving" if second_avg > first_avg + 0.03 else (
            "declining" if second_avg < first_avg - 0.03 else "stable"
        )
    else:
        trend = "insufficient data"

    habits_report = {
        "total_sessions": len(session_summaries),
        "total_prompts": total_prompts,
        "avg_score": round(avg_score, 4),
        "total_waste_from_prompts": total_waste,
        "top_antipatterns": top_antipatterns[:10],
        "trend": trend,
        "session_summaries": session_summaries,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  PROMPTING HABITS ACROSS {len(session_summaries)} SESSIONS")
        print(f"{'=' * 60}")
        print(f"  Total prompts: {total_prompts}")
        print(f"  Avg score: {avg_score * 100:.0f}/100")
        print(f"  Trend: {trend}")
        print(f"  Total waste from prompt quality: ~{total_waste} turns")
        if top_antipatterns:
            print(f"\n  Your biggest prompt habits to fix:")
            for name, count in top_antipatterns[:5]:
                ap = PROMPT_ANTIPATTERNS.get(name, {})
                print(f"    {count}× {name}: {ap.get('fix', '')}")

    return habits_report


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prompt Quality Analyzer — Evaluate YOUR prompting patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Session analysis
    session_parser = subparsers.add_parser("session", help="Analyze prompts in a session")
    session_parser.add_argument("--history", help="Path to HISTORY.jsonl")
    session_parser.add_argument("--session-dir", help="Path to project directory")
    session_parser.add_argument("--output", default="analysis/", help="Output directory")
    session_parser.add_argument("--no-ai", action="store_true", help="Local analysis only (no API calls)")
    session_parser.add_argument("--json", action="store_true")
    session_parser.add_argument("--model", default="claude-sonnet-4-20250514")
    session_parser.add_argument("--quiet", action="store_true")

    # Single prompt check
    check_parser = subparsers.add_parser("check", help="Score a single prompt before sending")
    check_parser.add_argument("prompt", help="The prompt text to evaluate")
    check_parser.add_argument("--context", default="", help="Context from prior prompt")
    check_parser.add_argument("--no-ai", action="store_true")
    check_parser.add_argument("--model", default="claude-sonnet-4-20250514")
    check_parser.add_argument("--json", action="store_true")

    # Cross-session habits
    habits_parser = subparsers.add_parser("habits", help="Analyze habits across sessions")
    habits_parser.add_argument("--dirs", nargs="+", required=True, help="Session directories")
    habits_parser.add_argument("--output", default="analysis/", help="Output directory")
    habits_parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "check":
        analyzer = PromptAnalyzer(
            model=getattr(args, "model", "claude-sonnet-4-20250514"),
            use_ai=not args.no_ai,
        )
        score = analyzer.check_single(args.prompt, context=args.context)
        if args.json:
            print(json.dumps(score.to_dict(), indent=2))
        return

    if args.command == "session":
        history_path = args.history
        if not history_path and args.session_dir:
            history_path = str(Path(args.session_dir) / "HISTORY.jsonl")
        if not history_path:
            parser.error("Provide --history or --session-dir")

        analyzer = PromptAnalyzer(
            model=args.model,
            use_ai=not args.no_ai,
        )
        analysis = analyzer.analyze_session(history_path, verbose=not args.quiet)

        # Save
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"prompt_analysis_{timestamp}.json"
        output_path.write_text(json.dumps(analysis.to_dict(), indent=2))

        if args.json:
            print(json.dumps(analysis.to_dict(), indent=2))
        else:
            print(analysis.summary())
            print(f"\nFull results saved to: {output_path}")
        return

    if args.command == "habits":
        report = analyze_habits(args.dirs)
        if args.json:
            print(json.dumps(report, indent=2))
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "habits_report.json").write_text(json.dumps(report, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
