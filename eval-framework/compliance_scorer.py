#!/usr/bin/env python3
"""
Compliance Scorer for Agent Plan Prompt v2
==========================================
Validates that an agent session's artifacts conform to all structural
contracts defined in the system prompt.

Usage:
    python compliance_scorer.py --session-dir ./my-session
    python compliance_scorer.py --session-dir ./my-session --verbose
    python compliance_scorer.py --session-dir ./my-session --json

Expected session directory structure:
    my-session/
    ├── PLAN.md
    ├── HISTORY.jsonl
    ├── TEST_QUEUE.md      (optional)
    ├── RETRO.md           (optional)
    ├── PLAYBOOK.md        (optional)
    └── transcript.jsonl   (optional — raw turn-by-turn agent output)
"""

import json
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# ──────────────────────────────────────────────
# Scoring Model
# ──────────────────────────────────────────────

@dataclass
class Check:
    """Single compliance check result."""
    name: str
    category: str
    passed: bool
    weight: float = 1.0
    detail: str = ""
    severity: str = "error"  # error | warning | info

    @property
    def score(self) -> float:
        return self.weight if self.passed else 0.0


@dataclass
class CategoryScore:
    """Aggregated score for a category of checks."""
    name: str
    checks: list = field(default_factory=list)

    @property
    def score(self) -> float:
        total_weight = sum(c.weight for c in self.checks)
        if total_weight == 0:
            return 0.0
        return sum(c.score for c in self.checks) / total_weight

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


@dataclass
class ComplianceReport:
    """Full compliance report for a session."""
    session_dir: str
    categories: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def add(self, check: Check):
        if check.category not in self.categories:
            self.categories[check.category] = CategoryScore(check.category)
        self.categories[check.category].checks.append(check)

    @property
    def overall_score(self) -> float:
        all_checks = []
        for cat in self.categories.values():
            all_checks.extend(cat.checks)
        total_weight = sum(c.weight for c in all_checks)
        if total_weight == 0:
            return 0.0
        return sum(c.score for c in all_checks) / total_weight

    @property
    def total_checks(self) -> int:
        return sum(len(cat.checks) for cat in self.categories.values())

    @property
    def total_passed(self) -> int:
        return sum(cat.passed for cat in self.categories.values())

    @property
    def total_failed(self) -> int:
        return sum(cat.failed for cat in self.categories.values())

    def to_dict(self) -> dict:
        return {
            "session_dir": self.session_dir,
            "overall_score": round(self.overall_score * 100, 1),
            "total_checks": self.total_checks,
            "passed": self.total_passed,
            "failed": self.total_failed,
            "categories": {
                name: {
                    "score": round(cat.score * 100, 1),
                    "passed": cat.passed,
                    "failed": cat.failed,
                    "checks": [
                        {
                            "name": c.name,
                            "passed": c.passed,
                            "severity": c.severity,
                            "detail": c.detail,
                        }
                        for c in cat.checks
                    ],
                }
                for name, cat in self.categories.items()
            },
            "errors": self.errors,
        }


# ──────────────────────────────────────────────
# Valid Process Tags (from the prompt)
# ──────────────────────────────────────────────

VALID_PROCESS_TAGS = {
    # Productive
    "clean-execution",
    "good-discovery",
    "course-correct",
    # Waste
    "dead-end",
    "redo",
    "yak-shave",
    "scope-creep",
    "vague-prompt",
    "misfire",
    # Neutral
    "planning",
    "debug",
    "refactor",
}

WASTE_TAGS = {"dead-end", "redo", "yak-shave", "scope-creep", "vague-prompt", "misfire"}
PROCESS_NOTE_REQUIRED_TAGS = {"dead-end", "redo", "vague-prompt", "misfire"}


# ──────────────────────────────────────────────
# PLAN.md Checks
# ──────────────────────────────────────────────

def check_plan(report: ComplianceReport, session_dir: Path):
    """Validate PLAN.md structure and content."""
    cat = "PLAN.md"
    plan_path = session_dir / "PLAN.md"

    # Existence
    if not plan_path.exists():
        report.add(Check("file_exists", cat, False, 3.0,
                         "PLAN.md not found — this is the primary navigation file",
                         "error"))
        return

    report.add(Check("file_exists", cat, True, 3.0, "PLAN.md present"))
    content = plan_path.read_text(encoding="utf-8")
    lines = content.strip().split("\n")

    # Required sections
    required_sections = {
        "Goal": {"weight": 2.0, "pattern": r"^#{1,2}\s+Goal"},
        "Current Milestone": {"weight": 2.0, "pattern": r"^#{1,2}\s+Current Milestone"},
        "Status": {"weight": 2.0, "pattern": r"^#{1,2}\s+Status"},
        "Next Steps": {"weight": 2.0, "pattern": r"^#{1,2}\s+Next Steps"},
        "Active Context": {"weight": 1.5, "pattern": r"^#{1,2}\s+Active Context"},
        "Completed Milestones": {"weight": 1.0, "pattern": r"^#{1,2}\s+Completed Milestones"},
        "Prompt History": {"weight": 1.5, "pattern": r"^#{1,2}\s+Prompt History"},
    }

    for section, spec in required_sections.items():
        found = any(re.match(spec["pattern"], line) for line in lines)
        report.add(Check(
            f"section_{section.lower().replace(' ', '_')}",
            cat, found, spec["weight"],
            f"Section '{section}' {'found' if found else 'MISSING'}",
            "error" if spec["weight"] >= 2.0 else "warning"
        ))

    # Goal specificity — should be 1-3 sentences, not empty
    goal_content = _extract_section(content, "Goal")
    if goal_content:
        sentences = [s.strip() for s in re.split(r'[.!?]+', goal_content) if s.strip()]
        goal_ok = 1 <= len(sentences) <= 5
        report.add(Check("goal_specificity", cat, goal_ok, 1.0,
                         f"Goal has {len(sentences)} sentence(s) (target 1-3)",
                         "warning"))
    else:
        report.add(Check("goal_specificity", cat, False, 1.0,
                         "Goal section is empty", "warning"))

    # Status is one line
    status_content = _extract_section(content, "Status")
    if status_content:
        status_lines = [l for l in status_content.strip().split("\n") if l.strip()]
        report.add(Check("status_one_line", cat, len(status_lines) <= 2, 1.0,
                         f"Status has {len(status_lines)} line(s) (should be ~1)",
                         "warning"))
    else:
        report.add(Check("status_one_line", cat, False, 1.0,
                         "Status section is empty", "warning"))

    # Next Steps are numbered
    next_steps = _extract_section(content, "Next Steps")
    if next_steps:
        numbered = re.findall(r'^\s*\d+[\.\)]\s+', next_steps, re.MULTILINE)
        checkboxed = re.findall(r'^\s*-\s*\[[ x]\]\s+', next_steps, re.MULTILINE)
        items = len(numbered) + len(checkboxed)
        report.add(Check("next_steps_populated", cat, items >= 1, 1.5,
                         f"Next Steps has {items} item(s) (target 5-10)",
                         "warning"))
        report.add(Check("next_steps_range", cat, 3 <= items <= 15, 0.5,
                         f"{items} items — {'within' if 3 <= items <= 15 else 'outside'} ideal 5-10 range",
                         "info"))

    # Prompt History table
    history_section = _extract_section(content, "Prompt History")
    if history_section:
        has_table = bool(re.search(r'\|\s*P-\d+\s*\|', history_section))
        report.add(Check("prompt_history_table", cat, has_table, 1.0,
                         "Prompt history table with P-IDs " + ("found" if has_table else "not found"),
                         "warning"))
        # Check table isn't too long (should be 10-15)
        p_ids = re.findall(r'\|\s*(P-\d+)\s*\|', history_section)
        if len(p_ids) > 20:
            report.add(Check("prompt_history_pruned", cat, False, 0.5,
                             f"Prompt history has {len(p_ids)} entries — should be trimmed to 10-15",
                             "info"))
        else:
            report.add(Check("prompt_history_pruned", cat, True, 0.5,
                             f"Prompt history has {len(p_ids)} entries"))


# ──────────────────────────────────────────────
# HISTORY.jsonl Checks
# ──────────────────────────────────────────────

def check_history(report: ComplianceReport, session_dir: Path):
    """Validate HISTORY.jsonl structure and content quality."""
    cat = "HISTORY.jsonl"
    path = session_dir / "HISTORY.jsonl"

    if not path.exists():
        report.add(Check("file_exists", cat, False, 3.0,
                         "HISTORY.jsonl not found", "error"))
        return

    report.add(Check("file_exists", cat, True, 3.0, "HISTORY.jsonl present"))

    entries = []
    parse_errors = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            parse_errors += 1

    report.add(Check("valid_json_lines", cat, parse_errors == 0, 2.0,
                     f"{parse_errors} JSON parse error(s)" if parse_errors else "All lines valid JSON",
                     "error" if parse_errors else "info"))

    if not entries:
        report.add(Check("has_entries", cat, False, 2.0, "No valid entries", "error"))
        return

    # Required fields
    required_fields = {"id", "type", "milestone", "summary", "tags", "timestamp"}
    response_fields = {"process_tag"}

    missing_field_count = 0
    for entry in entries:
        missing = required_fields - set(entry.keys())
        if missing:
            missing_field_count += 1

    report.add(Check("required_fields", cat,
                     missing_field_count == 0, 2.0,
                     f"{missing_field_count}/{len(entries)} entries missing required fields",
                     "error"))

    # ID format: P-XXX for prompts, R-XXX for responses
    bad_ids = []
    for entry in entries:
        eid = entry.get("id", "")
        etype = entry.get("type", "")
        if etype == "prompt" and not re.match(r'^P-\d+$', eid):
            bad_ids.append(eid)
        elif etype == "response" and not re.match(r'^R-\d+$', eid):
            bad_ids.append(eid)
    report.add(Check("id_format", cat, len(bad_ids) == 0, 1.5,
                     f"Bad IDs: {bad_ids[:5]}" if bad_ids else "All IDs correctly formatted",
                     "error"))

    # Paired entries: every P-XXX should have a matching R-XXX
    prompt_ids = {e["id"].replace("P-", "") for e in entries if e.get("type") == "prompt"}
    response_ids = {e["id"].replace("R-", "") for e in entries if e.get("type") == "response"}
    unpaired_prompts = prompt_ids - response_ids
    unpaired_responses = response_ids - prompt_ids
    all_paired = len(unpaired_prompts) == 0 and len(unpaired_responses) == 0
    detail = "All prompt/response pairs matched"
    if not all_paired:
        parts = []
        if unpaired_prompts:
            parts.append(f"Prompts without responses: P-{', P-'.join(sorted(unpaired_prompts)[:5])}")
        if unpaired_responses:
            parts.append(f"Responses without prompts: R-{', R-'.join(sorted(unpaired_responses)[:5])}")
        detail = "; ".join(parts)
    report.add(Check("paired_entries", cat, all_paired, 1.5, detail, "warning"))

    # Process tags on all responses
    responses = [e for e in entries if e.get("type") == "response"]
    missing_tags = [e["id"] for e in responses if "process_tag" not in e or not e["process_tag"]]
    report.add(Check("process_tags_present", cat,
                     len(missing_tags) == 0, 2.0,
                     f"{len(missing_tags)} response(s) missing process_tag" if missing_tags
                     else "All responses have process_tag",
                     "error"))

    # Process tags use valid vocabulary
    invalid_tags = []
    for e in responses:
        tag = e.get("process_tag", "")
        if tag and tag not in VALID_PROCESS_TAGS:
            invalid_tags.append((e.get("id", "?"), tag))
    report.add(Check("process_tags_valid", cat,
                     len(invalid_tags) == 0, 1.5,
                     f"Invalid tags: {invalid_tags[:5]}" if invalid_tags
                     else "All process tags from valid vocabulary",
                     "error"))

    # Waste tags require process_note
    missing_notes = []
    for e in responses:
        tag = e.get("process_tag", "")
        if tag in PROCESS_NOTE_REQUIRED_TAGS and not e.get("process_note"):
            missing_notes.append(e.get("id", "?"))
    report.add(Check("waste_tags_have_notes", cat,
                     len(missing_notes) == 0, 1.5,
                     f"Waste entries missing process_note: {missing_notes[:5]}" if missing_notes
                     else "All waste-tagged entries have process_note",
                     "error"))

    # Summary quality — check for vague summaries
    vague_patterns = [
        r'^(updated|fixed|changed|worked on|did)\b',
        r'^(stuff|things|misc|various)\b',
    ]
    vague_summaries = []
    for e in entries:
        summary = e.get("summary", "").lower().strip()
        if any(re.match(p, summary) for p in vague_patterns):
            vague_summaries.append((e.get("id", "?"), summary))
    report.add(Check("summary_specificity", cat,
                     len(vague_summaries) == 0, 1.0,
                     f"Vague summaries found: {vague_summaries[:3]}" if vague_summaries
                     else "All summaries appear specific",
                     "warning"))

    # Tags field is a list of 2-5 items
    bad_tag_counts = []
    for e in entries:
        tags = e.get("tags", [])
        if not isinstance(tags, list) or not (1 <= len(tags) <= 7):
            bad_tag_counts.append((e.get("id", "?"), len(tags) if isinstance(tags, list) else "not a list"))
    report.add(Check("tags_count", cat,
                     len(bad_tag_counts) == 0, 0.5,
                     f"Entries with bad tag counts: {bad_tag_counts[:5]}" if bad_tag_counts
                     else "All entries have 2-5 tags",
                     "info"))

    # Timestamps are valid ISO 8601
    bad_timestamps = []
    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                bad_timestamps.append((e.get("id", "?"), ts))
        else:
            bad_timestamps.append((e.get("id", "?"), "<missing>"))
    report.add(Check("timestamps_valid", cat,
                     len(bad_timestamps) == 0, 1.0,
                     f"Bad timestamps: {bad_timestamps[:5]}" if bad_timestamps
                     else "All timestamps valid ISO 8601",
                     "warning"))

    # Sequential IDs (no gaps or duplicates)
    prompt_nums = sorted(int(e["id"].replace("P-", ""))
                         for e in entries if e.get("type") == "prompt"
                         and re.match(r'^P-\d+$', e.get("id", "")))
    if prompt_nums:
        expected = list(range(prompt_nums[0], prompt_nums[-1] + 1))
        gaps = set(expected) - set(prompt_nums)
        dupes = len(prompt_nums) - len(set(prompt_nums))
        seq_ok = len(gaps) == 0 and dupes == 0
        detail = "Sequential IDs OK"
        if gaps:
            detail = f"ID gaps: {sorted(gaps)[:10]}"
        if dupes:
            detail += f"; {dupes} duplicate(s)"
        report.add(Check("sequential_ids", cat, seq_ok, 0.5, detail, "info"))


# ──────────────────────────────────────────────
# TEST_QUEUE.md Checks
# ──────────────────────────────────────────────

def check_test_queue(report: ComplianceReport, session_dir: Path):
    """Validate TEST_QUEUE.md structure."""
    cat = "TEST_QUEUE.md"
    path = session_dir / "TEST_QUEUE.md"

    if not path.exists():
        report.add(Check("file_exists", cat, True, 0.5,
                         "TEST_QUEUE.md not present (optional)", "info"))
        return

    report.add(Check("file_exists", cat, True, 1.0, "TEST_QUEUE.md present"))
    content = path.read_text(encoding="utf-8")

    # Has section headers (Ready to Write / Deferred)
    has_ready = bool(re.search(r'#{1,3}\s+Ready', content, re.IGNORECASE))
    has_deferred = bool(re.search(r'#{1,3}\s+Deferred', content, re.IGNORECASE))
    report.add(Check("sections", cat, has_ready, 1.0,
                     f"Ready section {'found' if has_ready else 'missing'}; "
                     f"Deferred section {'found' if has_deferred else 'missing'}",
                     "warning"))

    # TQ-XXX IDs present
    tq_ids = re.findall(r'TQ-\d+', content)
    report.add(Check("tq_ids", cat, len(tq_ids) > 0, 1.0,
                     f"{len(tq_ids)} TQ entries found",
                     "warning" if not tq_ids else "info"))

    # Breadcrumb quality — check for required sub-fields
    entries = re.split(r'###\s+TQ-\d+', content)[1:]  # split on TQ headers
    quality_issues = []
    required_subfields = ["What", "Happy path"]
    desirable_subfields = ["Error states", "Edge cases", "Selectors", "Setup"]

    for i, entry in enumerate(entries):
        for sf in required_subfields:
            if sf.lower() not in entry.lower():
                quality_issues.append(f"TQ entry {i + 1} missing '{sf}'")

    desirable_present = 0
    desirable_total = 0
    for entry in entries:
        for sf in desirable_subfields:
            desirable_total += 1
            if sf.lower() in entry.lower():
                desirable_present += 1

    if entries:
        report.add(Check("breadcrumb_required_fields", cat,
                         len(quality_issues) == 0, 1.5,
                         "; ".join(quality_issues[:5]) if quality_issues
                         else "All entries have required subfields",
                         "warning"))
        if desirable_total > 0:
            ratio = desirable_present / desirable_total
            report.add(Check("breadcrumb_completeness", cat,
                             ratio >= 0.5, 1.0,
                             f"Desirable subfields present: {desirable_present}/{desirable_total} ({ratio:.0%})",
                             "info"))


# ──────────────────────────────────────────────
# RETRO.md Checks
# ──────────────────────────────────────────────

def check_retro(report: ComplianceReport, session_dir: Path):
    """Validate RETRO.md structure."""
    cat = "RETRO.md"
    path = session_dir / "RETRO.md"

    if not path.exists():
        report.add(Check("file_exists", cat, True, 0.0,
                         "RETRO.md not present (written at milestone boundaries)", "info"))
        return

    report.add(Check("file_exists", cat, True, 1.0, "RETRO.md present"))
    content = path.read_text(encoding="utf-8")

    # Required sections
    retro_sections = {
        "Metrics": r"#{1,3}\s+Metrics",
        "Optimal Path": r"#{1,3}\s+Optimal Path",
        "Key Waste Analysis": r"#{1,3}\s+(Key )?Waste",
        "Playbook Candidates": r"#{1,3}\s+Playbook",
    }

    for section, pattern in retro_sections.items():
        found = bool(re.search(pattern, content, re.IGNORECASE))
        report.add(Check(f"section_{section.lower().replace(' ', '_')}", cat,
                         found, 1.5,
                         f"'{section}' section {'found' if found else 'MISSING'}",
                         "warning"))

    # Metrics include counts
    metrics_section = _extract_section(content, "Metrics")
    if metrics_section:
        has_counts = bool(re.search(r'\d+\s*\(?\d*%?\)?', metrics_section))
        report.add(Check("metrics_quantified", cat, has_counts, 1.0,
                         "Metrics section contains quantified data" if has_counts
                         else "Metrics section lacks quantified data",
                         "warning"))

    # Optimal path includes estimated turns
    optimal = _extract_section(content, "Optimal Path")
    if optimal:
        has_estimate = bool(re.search(r'(estimated|optimal|~?\d+)\s*(turns|prompts)', optimal, re.IGNORECASE))
        report.add(Check("optimal_path_estimate", cat, has_estimate, 1.0,
                         "Optimal path includes turn estimate" if has_estimate
                         else "Optimal path missing turn estimate",
                         "warning"))

    # Prompt range specified
    has_range = bool(re.search(r'P-\d+\s*[→→\-–]\s*P-\d+', content))
    report.add(Check("prompt_range", cat, has_range, 0.5,
                     "Prompt range specified" if has_range else "Prompt range not found",
                     "info"))


# ──────────────────────────────────────────────
# Every-Turn Protocol Checks (from transcript)
# ──────────────────────────────────────────────

def check_turn_protocol(report: ComplianceReport, session_dir: Path):
    """Validate the every-turn Done/Next/Process footer in transcript."""
    cat = "Every-Turn Protocol"
    path = session_dir / "transcript.jsonl"

    if not path.exists():
        report.add(Check("transcript_available", cat, False, 0.0,
                         "transcript.jsonl not found — skipping turn protocol checks",
                         "info"))
        return

    report.add(Check("transcript_available", cat, True, 0.5, "Transcript available"))

    turns = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    agent_turns = [t for t in turns if t.get("role") == "assistant"]
    if not agent_turns:
        report.add(Check("has_agent_turns", cat, False, 1.0,
                         "No agent turns in transcript", "warning"))
        return

    # Check for Done/Next/Process footer
    footer_pattern = re.compile(
        r'\*\*Done:?\*\*.*\n.*\*\*Next:?\*\*.*\n.*\*\*Process:?\*\*',
        re.IGNORECASE | re.DOTALL
    )
    # Also accept markdown-free version
    footer_alt = re.compile(
        r'Done:.*\nNext:.*\nProcess:',
        re.IGNORECASE
    )

    turns_with_footer = 0
    turns_missing_footer = []
    for i, turn in enumerate(agent_turns):
        content = turn.get("content", "")
        if footer_pattern.search(content) or footer_alt.search(content):
            turns_with_footer += 1
        else:
            turns_missing_footer.append(i + 1)

    ratio = turns_with_footer / len(agent_turns) if agent_turns else 0
    report.add(Check("footer_compliance", cat, ratio >= 0.9, 2.0,
                     f"{turns_with_footer}/{len(agent_turns)} turns have Done/Next/Process footer ({ratio:.0%})",
                     "error" if ratio < 0.7 else "warning"))

    # Check that Process line contains a valid tag
    process_tags_found = 0
    for turn in agent_turns:
        content = turn.get("content", "")
        match = re.search(r'\*\*Process:?\*\*\s*\[?`?(\S+?)`?\]?', content)
        if not match:
            match = re.search(r'Process:\s*\[?`?(\S+?)`?\]?', content)
        if match:
            tag = match.group(1).strip("`[]")
            if tag in VALID_PROCESS_TAGS:
                process_tags_found += 1

    if agent_turns:
        tag_ratio = process_tags_found / len(agent_turns)
        report.add(Check("footer_valid_tags", cat, tag_ratio >= 0.8, 1.0,
                         f"{process_tags_found}/{len(agent_turns)} footers have valid process tags ({tag_ratio:.0%})",
                         "warning"))


# ──────────────────────────────────────────────
# Cross-File Consistency
# ──────────────────────────────────────────────

def check_consistency(report: ComplianceReport, session_dir: Path):
    """Check cross-file consistency."""
    cat = "Cross-File Consistency"

    plan_path = session_dir / "PLAN.md"
    history_path = session_dir / "HISTORY.jsonl"

    if not plan_path.exists() or not history_path.exists():
        report.add(Check("files_present", cat, False, 0.0,
                         "Need both PLAN.md and HISTORY.jsonl for consistency checks", "info"))
        return

    plan_content = plan_path.read_text(encoding="utf-8")
    history_entries = []
    for line in history_path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            history_entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # PLAN prompt history IDs should match HISTORY.jsonl
    plan_ids = set(re.findall(r'P-(\d+)', plan_content))
    history_ids = {e["id"].replace("P-", "").replace("R-", "")
                   for e in history_entries if e.get("id")}

    if plan_ids and history_ids:
        plan_max = max(int(x) for x in plan_ids) if plan_ids else 0
        hist_max = max(int(x) for x in history_ids) if history_ids else 0
        # Latest PLAN ID should be close to latest HISTORY ID
        id_gap = abs(plan_max - hist_max)
        report.add(Check("id_sync", cat, id_gap <= 2, 1.0,
                         f"PLAN latest P-{plan_max}, HISTORY latest {hist_max} (gap: {id_gap})",
                         "warning"))

    # Current milestone in PLAN matches milestone tags in recent HISTORY entries
    milestone_match = re.search(r'#{1,2}\s+Current Milestone\s*\n(.+?)(?:\n#|\Z)',
                                plan_content, re.DOTALL)
    if milestone_match and history_entries:
        plan_milestone = milestone_match.group(1).strip().split("\n")[0].strip()
        recent_milestones = {e.get("milestone", "") for e in history_entries[-10:]}
        # Fuzzy check — at least one recent entry should reference the current milestone
        any_match = any(
            plan_milestone.lower() in m.lower() or m.lower() in plan_milestone.lower()
            for m in recent_milestones if m
        )
        report.add(Check("milestone_sync", cat, any_match, 1.0,
                         f"PLAN milestone '{plan_milestone}' "
                         f"{'matches' if any_match else 'not found in'} recent HISTORY entries",
                         "warning"))


# ──────────────────────────────────────────────
# Process Health Metrics
# ──────────────────────────────────────────────

def check_process_health(report: ComplianceReport, session_dir: Path):
    """Compute process health metrics from HISTORY.jsonl."""
    cat = "Process Health"
    path = session_dir / "HISTORY.jsonl"

    if not path.exists():
        return

    entries = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    responses = [e for e in entries if e.get("type") == "response" and e.get("process_tag")]
    if len(responses) < 5:
        report.add(Check("sufficient_data", cat, True, 0.0,
                         f"Only {len(responses)} tagged responses — metrics may not be meaningful",
                         "info"))
        return

    # Tally tags
    productive = sum(1 for r in responses if r["process_tag"] in
                     {"clean-execution", "good-discovery", "course-correct"})
    waste = sum(1 for r in responses if r["process_tag"] in WASTE_TAGS)
    neutral = sum(1 for r in responses if r["process_tag"] in
                  {"planning", "debug", "refactor"})
    total = len(responses)

    prod_ratio = productive / total
    waste_ratio = waste / total

    # Playbook targets: productive 70%+, waste <15%
    report.add(Check("productive_ratio", cat, prod_ratio >= 0.60, 1.5,
                     f"Productive: {productive}/{total} ({prod_ratio:.0%}) — target ≥70%",
                     "warning" if prod_ratio < 0.70 else "info"))
    report.add(Check("waste_ratio", cat, waste_ratio <= 0.20, 1.5,
                     f"Waste: {waste}/{total} ({waste_ratio:.0%}) — target <15%",
                     "warning" if waste_ratio > 0.15 else "info"))

    # Check for 3+ consecutive waste turns (should trigger mandatory planning)
    consecutive_waste = 0
    max_consecutive = 0
    for r in responses:
        if r["process_tag"] in WASTE_TAGS:
            consecutive_waste += 1
            max_consecutive = max(max_consecutive, consecutive_waste)
        else:
            consecutive_waste = 0

    report.add(Check("waste_streak", cat, max_consecutive < 3, 1.0,
                     f"Max consecutive waste turns: {max_consecutive} "
                     f"(3+ should trigger mandatory planning)",
                     "warning" if max_consecutive >= 3 else "info"))

    # Tag honesty heuristic: if >90% is clean-execution, suspicious
    ce_count = sum(1 for r in responses if r["process_tag"] == "clean-execution")
    ce_ratio = ce_count / total if total else 0
    report.add(Check("tag_diversity", cat, ce_ratio < 0.90, 0.5,
                     f"clean-execution: {ce_ratio:.0%} of all tags "
                     f"({'suspiciously high — may indicate flattering tagging' if ce_ratio >= 0.90 else 'reasonable distribution'})",
                     "warning" if ce_ratio >= 0.90 else "info"))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _extract_section(content: str, heading: str) -> Optional[str]:
    """Extract content under a markdown heading (stops at next heading of same or higher level)."""
    pattern = re.compile(
        rf'^(#{1,3})\s+{re.escape(heading)}[^\n]*\n(.*?)(?=^\1\s+|\Z)',
        re.MULTILINE | re.DOTALL
    )
    # Try simpler pattern if the above doesn't match
    match = pattern.search(content)
    if match:
        return match.group(2).strip()

    # Fallback: just grab content between this heading and the next
    lines = content.split("\n")
    capturing = False
    captured = []
    heading_level = 0
    for line in lines:
        heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
        if heading_match:
            if heading.lower() in heading_match.group(2).lower():
                capturing = True
                heading_level = len(heading_match.group(1))
                continue
            elif capturing and len(heading_match.group(1)) <= heading_level:
                break
        if capturing:
            captured.append(line)
    return "\n".join(captured).strip() if captured else None


# ──────────────────────────────────────────────
# Report Rendering
# ──────────────────────────────────────────────

def render_terminal(report: ComplianceReport, verbose: bool = False):
    """Render report to terminal with color."""
    PASS = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    WARN = "\033[93m!\033[0m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIM = "\033[2m"

    score = report.overall_score * 100

    # Header
    print()
    print(f"{'─' * 60}")
    print(f"  {BOLD}COMPLIANCE SCORE: {score:.0f}/100{RESET}")
    print(f"  {report.total_passed} passed · {report.total_failed} failed · {report.total_checks} total")
    print(f"{'─' * 60}")

    for cat_name, cat in report.categories.items():
        cat_score = cat.score * 100
        icon = PASS if cat_score >= 90 else (WARN if cat_score >= 60 else FAIL)
        print(f"\n  {icon} {BOLD}{cat_name}{RESET}  {cat_score:.0f}%")

        if verbose or cat_score < 100:
            for check in cat.checks:
                if verbose or not check.passed:
                    status = PASS if check.passed else (FAIL if check.severity == "error" else WARN)
                    print(f"    {status} {check.name}")
                    if check.detail and (verbose or not check.passed):
                        print(f"      {DIM}{check.detail}{RESET}")

    print(f"\n{'─' * 60}")

    if report.errors:
        print(f"\n  {FAIL} Errors:")
        for err in report.errors:
            print(f"    · {err}")
        print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_compliance_check(session_dir: str, verbose: bool = False, json_output: bool = False):
    """Run all compliance checks and return report."""
    session_path = Path(session_dir)

    if not session_path.exists():
        print(f"Error: Session directory '{session_dir}' not found.")
        sys.exit(1)

    report = ComplianceReport(session_dir=str(session_path))

    # Run all check suites
    check_plan(report, session_path)
    check_history(report, session_path)
    check_test_queue(report, session_path)
    check_retro(report, session_path)
    check_turn_protocol(report, session_path)
    check_consistency(report, session_path)
    check_process_health(report, session_path)

    # Output
    if json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        render_terminal(report, verbose)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compliance Scorer for Agent Plan Prompt v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compliance_scorer.py --session-dir ./my-session
  python compliance_scorer.py --session-dir ./my-session --verbose
  python compliance_scorer.py --session-dir ./my-session --json
  python compliance_scorer.py --session-dir ./my-session --json > report.json
        """
    )
    parser.add_argument("--session-dir", required=True,
                        help="Path to session directory containing PLAN.md, HISTORY.jsonl, etc.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all checks, not just failures")
    parser.add_argument("--json", "-j", action="store_true", dest="json_output",
                        help="Output as JSON instead of terminal report")

    args = parser.parse_args()
    report = run_compliance_check(args.session_dir, args.verbose, args.json_output)

    # Exit code: 0 if score >= 80, 1 otherwise
    sys.exit(0 if report.overall_score >= 0.80 else 1)


if __name__ == "__main__":
    main()
