#!/usr/bin/env python3
"""
Task Runner — Executes synthetic tasks through prompt variants and collects metrics.

Runs tasks from task_bank.json through different system prompt configurations,
simulating multi-turn agent interactions and measuring:
- Task completion (done criteria met)
- Turn efficiency (actual vs optimal)
- Waste ratio (process tag distribution)
- Context recovery time (for session-start tasks)

Usage:
    python runner.py --variant full --tasks tasks/task_bank.json --output results/
    python runner.py --variant minimal --tasks tasks/task_bank.json --output results/
    python runner.py --variant none --tasks tasks/task_bank.json --output results/
    python runner.py --variant full --task-id T-001 --output results/  # Single task
    python runner.py --compare results/full/ results/minimal/ results/none/

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


# ── Prompt Variants ─────────────────────────────────────────────────────────

VARIANT_FULL = "full"          # Complete agent-plan-prompt-v2
VARIANT_MINIMAL = "minimal"    # PLAN.md + HISTORY.jsonl only, no process tags, no test breadcrumbs
VARIANT_NONE = "none"          # No system prompt
VARIANT_CUSTOM = "custom"      # User-provided system prompt file

MINIMAL_SYSTEM_PROMPT = """You are a coding agent. Maintain two files:

PLAN.md — Track what you're building, current status, and next steps.
HISTORY.jsonl — Log each prompt/response with an ID, summary, and tags.

After every turn, update both files and tell the user what you did and what's next."""

def load_full_system_prompt(path: str = None) -> str:
    """Load the full v2 system prompt."""
    if path:
        return Path(path).read_text()
    # Default: look for the prompt file in common locations
    candidates = [
        Path(__file__).parent.parent / "agent-plan-prompt-v2.md",
        Path("agent-plan-prompt-v2.md"),
        Path.home() / "agent-plan-prompt-v2.md",
    ]
    for c in candidates:
        if c.exists():
            return c.read_text()
    raise FileNotFoundError(
        "Could not find agent-plan-prompt-v2.md. "
        "Provide path with --system-prompt or place in project root."
    )


# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn_number: int
    prompt: str
    response: str
    latency_ms: int
    token_count_prompt: int
    token_count_response: int
    process_tag: Optional[str] = None  # Extracted from response if present
    has_footer: bool = False
    plan_updated: bool = False
    history_appended: bool = False

@dataclass
class TaskResult:
    task_id: str
    task_name: str
    variant: str
    turns: list[TurnResult] = field(default_factory=list)
    completed: bool = False
    done_criteria_met: list[str] = field(default_factory=list)
    done_criteria_missed: list[str] = field(default_factory=list)
    total_turns: int = 0
    optimal_turns: int = 0
    start_time: str = ""
    end_time: str = ""
    error: Optional[str] = None

    @property
    def efficiency_ratio(self) -> float:
        if self.optimal_turns == 0:
            return 0.0
        return self.total_turns / self.optimal_turns

    @property
    def completion_rate(self) -> float:
        total = len(self.done_criteria_met) + len(self.done_criteria_missed)
        if total == 0:
            return 0.0
        return len(self.done_criteria_met) / total

    @property
    def waste_ratio(self) -> float:
        waste_tags = {"dead-end", "redo", "yak-shave", "scope-creep", "vague-prompt", "misfire"}
        tagged = [t for t in self.turns if t.process_tag]
        if not tagged:
            return 0.0
        waste = sum(1 for t in tagged if t.process_tag in waste_tags)
        return waste / len(tagged)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "variant": self.variant,
            "completed": self.completed,
            "completion_rate": round(self.completion_rate, 4),
            "total_turns": self.total_turns,
            "optimal_turns": self.optimal_turns,
            "efficiency_ratio": round(self.efficiency_ratio, 2),
            "waste_ratio": round(self.waste_ratio, 4),
            "done_criteria_met": self.done_criteria_met,
            "done_criteria_missed": self.done_criteria_missed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
            "turns": [asdict(t) for t in self.turns],
        }


@dataclass
class VariantReport:
    variant: str
    task_results: list[TaskResult] = field(default_factory=list)

    @property
    def avg_completion_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.completion_rate for r in self.task_results) / len(self.task_results)

    @property
    def avg_efficiency_ratio(self) -> float:
        completed = [r for r in self.task_results if r.completed]
        if not completed:
            return 0.0
        return sum(r.efficiency_ratio for r in completed) / len(completed)

    @property
    def avg_waste_ratio(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(r.waste_ratio for r in self.task_results) / len(self.task_results)

    def summary(self) -> str:
        lines = [
            f"Variant: {self.variant}",
            f"Tasks run: {len(self.task_results)}",
            f"Avg completion: {self.avg_completion_rate:.1%}",
            f"Avg efficiency ratio: {self.avg_efficiency_ratio:.1f}x optimal",
            f"Avg waste ratio: {self.avg_waste_ratio:.1%}",
        ]
        return "\n".join(lines)


# ── Task Execution Engine ──────────────────────────────────────────────────

class TaskRunner:
    """Executes tasks through the Anthropic API with a given system prompt variant."""

    def __init__(
        self,
        variant: str,
        system_prompt: str = "",
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 40,
        api_key: Optional[str] = None,
    ):
        self.variant = variant
        self.system_prompt = system_prompt
        self.model = model
        self.max_turns = max_turns
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required. Set env var or pass api_key.")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            print("Install anthropic: pip install anthropic --break-system-packages")
            sys.exit(1)

    def _build_initial_prompt(self, task: dict) -> str:
        """Build the first user message from task definition."""
        parts = [f"Build the following: {task['name']}", "", task["description"], ""]

        if task.get("constraints"):
            parts.append("Constraints:")
            for k, v in task["constraints"].items():
                parts.append(f"  - {k}: {v}")
            parts.append("")

        parts.append("Done criteria:")
        for c in task.get("done_criteria", []):
            parts.append(f"  - {c}")

        return "\n".join(parts)

    def _build_followup_prompt(self, task: dict, turn: int, last_response: str) -> str:
        """Build follow-up prompts for multi-turn tasks."""
        # Simple continuation — real A/B testing would use scripted sequences
        return "Continue with the next step from your plan."

    def _extract_process_tag(self, response: str) -> Optional[str]:
        """Extract process tag from response footer."""
        import re
        match = re.search(r"\*\*Process:\*\*\s*`?(\S+?)`?\s", response)
        if match:
            tag = match.group(1).strip("`*")
            return tag
        return None

    def _check_footer(self, response: str) -> bool:
        """Check if response has the Done/Next/Process footer."""
        import re
        return bool(re.search(
            r"\*\*Done:\*\*.*\*\*Next:\*\*.*\*\*Process:\*\*",
            response,
            re.DOTALL,
        ))

    def _evaluate_completion(self, task: dict, conversation: list[dict]) -> tuple[list[str], list[str]]:
        """Use a judge call to evaluate which done criteria were met."""
        criteria = task.get("done_criteria", [])
        if not criteria:
            return [], []

        # Build evaluation prompt
        conv_text = "\n\n".join(
            f"{'USER' if m['role'] == 'user' else 'AGENT'}: {m['content'][:2000]}"
            for m in conversation
        )

        eval_prompt = f"""Evaluate whether each of these done criteria was met in the following conversation.

DONE CRITERIA:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(criteria))}

CONVERSATION (truncated):
{conv_text[:8000]}

Respond with ONLY a JSON object: {{"met": [1, 3, 5], "missed": [2, 4]}} where numbers are the criteria indices (1-based).
No explanation, just JSON."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": eval_prompt}],
            )
            text = response.content[0].text.strip()
            # Clean potential markdown fencing
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            met = [criteria[i - 1] for i in result.get("met", []) if 1 <= i <= len(criteria)]
            missed = [criteria[i - 1] for i in result.get("missed", []) if 1 <= i <= len(criteria)]
            return met, missed
        except Exception as e:
            print(f"  ⚠ Completion evaluation failed: {e}")
            return [], criteria

    def run_task(self, task: dict, verbose: bool = True) -> TaskResult:
        """Execute a single task and return results."""
        result = TaskResult(
            task_id=task["id"],
            task_name=task["name"],
            variant=self.variant,
            optimal_turns=task.get("optimal_turns", 0),
            start_time=datetime.now().isoformat(),
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task['id']} — {task['name']} (tier: {task.get('tier', '?')})")
            print(f"Variant: {self.variant} | Optimal turns: {task.get('optimal_turns', '?')}")
            print(f"{'='*60}")

        conversation = []
        initial_prompt = self._build_initial_prompt(task)

        # Turn loop
        max_turns_for_task = min(
            self.max_turns,
            task.get("optimal_turns", 10) * 3,  # Cap at 3x optimal
        )

        for turn_num in range(1, max_turns_for_task + 1):
            if turn_num == 1:
                user_msg = initial_prompt
            else:
                user_msg = self._build_followup_prompt(task, turn_num, conversation[-1]["content"])

            conversation.append({"role": "user", "content": user_msg})

            if verbose:
                print(f"\n  Turn {turn_num}: {user_msg[:80]}...")

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

                turn_result = TurnResult(
                    turn_number=turn_num,
                    prompt=user_msg[:500],
                    response=response_text[:500],
                    latency_ms=latency,
                    token_count_prompt=response.usage.input_tokens,
                    token_count_response=response.usage.output_tokens,
                    process_tag=self._extract_process_tag(response_text),
                    has_footer=self._check_footer(response_text),
                    plan_updated="PLAN.md" in response_text or "## Status" in response_text,
                    history_appended="HISTORY.jsonl" in response_text or "P-" in response_text,
                )
                result.turns.append(turn_result)

                if verbose:
                    tag_str = f" [{turn_result.process_tag}]" if turn_result.process_tag else ""
                    print(f"  → {latency}ms, {response.usage.output_tokens} tokens{tag_str}")

            except Exception as e:
                result.error = str(e)
                if verbose:
                    print(f"  ✗ Error: {e}")
                break

        result.total_turns = len(result.turns)
        result.end_time = datetime.now().isoformat()

        # Evaluate completion
        if verbose:
            print(f"\n  Evaluating completion...")

        met, missed = self._evaluate_completion(task, conversation)
        result.done_criteria_met = met
        result.done_criteria_missed = missed
        result.completed = len(missed) == 0 and len(met) > 0

        if verbose:
            print(f"  ✓ Met: {len(met)}/{len(met) + len(missed)} criteria")
            print(f"  Efficiency: {result.efficiency_ratio:.1f}x optimal")
            print(f"  Waste: {result.waste_ratio:.0%}")

        return result


# ── Comparison Engine ──────────────────────────────────────────────────────

def compare_variants(result_dirs: list[str]) -> str:
    """Compare results across variants and produce a comparison report."""
    variants = {}

    for d in result_dirs:
        path = Path(d)
        for f in path.glob("*.json"):
            data = json.loads(f.read_text())
            variant = data.get("variant", path.name)
            if variant not in variants:
                variants[variant] = []
            variants[variant].append(data)

    lines = []
    lines.append("=" * 70)
    lines.append("VARIANT COMPARISON REPORT")
    lines.append("=" * 70)

    # Summary table
    lines.append(f"\n{'Variant':<15} {'Tasks':>6} {'Completed':>10} {'Avg Eff.':>10} {'Avg Waste':>10}")
    lines.append("-" * 55)

    for variant, results in sorted(variants.items()):
        n = len(results)
        completed = sum(1 for r in results if r.get("completed"))
        avg_eff = sum(r.get("efficiency_ratio", 0) for r in results if r.get("completed")) / max(completed, 1)
        avg_waste = sum(r.get("waste_ratio", 0) for r in results) / max(n, 1)
        lines.append(f"{variant:<15} {n:>6} {completed:>10} {avg_eff:>9.1f}x {avg_waste:>9.1%}")

    # Per-task comparison
    all_task_ids = sorted({r["task_id"] for results in variants.values() for r in results})

    lines.append(f"\n\nPER-TASK BREAKDOWN")
    lines.append("-" * 70)

    for task_id in all_task_ids:
        lines.append(f"\n{task_id}:")
        for variant, results in sorted(variants.items()):
            task_results = [r for r in results if r["task_id"] == task_id]
            if task_results:
                r = task_results[0]
                status = "✅" if r.get("completed") else "❌"
                lines.append(
                    f"  {variant:<12} {status} "
                    f"turns={r.get('total_turns', '?')}/{r.get('optimal_turns', '?')} "
                    f"eff={r.get('efficiency_ratio', 0):.1f}x "
                    f"waste={r.get('waste_ratio', 0):.0%} "
                    f"criteria={len(r.get('done_criteria_met', []))}/{len(r.get('done_criteria_met', [])) + len(r.get('done_criteria_missed', []))}"
                )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Task Runner for Agent Plan Prompt v2 Eval")
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run tasks through a variant")
    run_parser.add_argument("--variant", choices=["full", "minimal", "none", "custom"], required=True)
    run_parser.add_argument("--tasks", default="tasks/task_bank.json", help="Path to task bank")
    run_parser.add_argument("--task-id", help="Run only this task ID")
    run_parser.add_argument("--tier", help="Run only tasks of this tier")
    run_parser.add_argument("--output", default="results/", help="Output directory")
    run_parser.add_argument("--system-prompt", help="Path to system prompt (for custom variant)")
    run_parser.add_argument("--model", default="claude-sonnet-4-20250514")
    run_parser.add_argument("--max-turns", type=int, default=40)
    run_parser.add_argument("--quiet", action="store_true")

    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare variant results")
    cmp_parser.add_argument("result_dirs", nargs="+", help="Directories with result JSON files")

    # List command
    list_parser = subparsers.add_parser("list", help="List available tasks")
    list_parser.add_argument("--tasks", default="tasks/task_bank.json")

    args = parser.parse_args()

    if args.command == "list":
        tasks = json.loads(Path(args.tasks).read_text())["tasks"]
        for t in tasks:
            print(f"  {t['id']:<8} [{t.get('tier', '?'):<12}] {t['name']:<35} (optimal: {t.get('optimal_turns', '?')} turns)")
        return

    if args.command == "compare":
        print(compare_variants(args.result_dirs))
        return

    if args.command == "run":
        # Load tasks
        task_data = json.loads(Path(args.tasks).read_text())
        tasks = task_data["tasks"]

        # Filter
        if args.task_id:
            tasks = [t for t in tasks if t["id"] == args.task_id]
        if args.tier:
            tasks = [t for t in tasks if t.get("tier") == args.tier]

        if not tasks:
            print("No tasks matched filters.")
            return

        # Load system prompt
        if args.variant == "full":
            system_prompt = load_full_system_prompt(args.system_prompt)
        elif args.variant == "minimal":
            system_prompt = MINIMAL_SYSTEM_PROMPT
        elif args.variant == "custom":
            if not args.system_prompt:
                parser.error("--system-prompt required for custom variant")
            system_prompt = Path(args.system_prompt).read_text()
        else:
            system_prompt = ""

        # Create runner
        runner = TaskRunner(
            variant=args.variant,
            system_prompt=system_prompt,
            model=args.model,
            max_turns=args.max_turns,
        )

        # Output directory
        output_dir = Path(args.output) / args.variant
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running {len(tasks)} tasks with variant '{args.variant}'")
        print(f"Output: {output_dir}")

        results = []
        for task in tasks:
            result = runner.run_task(task, verbose=not args.quiet)
            results.append(result)

            # Save individual result
            result_path = output_dir / f"{task['id']}.json"
            result_path.write_text(json.dumps(result.to_dict(), indent=2))

        # Save summary
        summary = {
            "variant": args.variant,
            "model": args.model,
            "tasks_run": len(results),
            "completed": sum(1 for r in results if r.completed),
            "avg_completion_rate": sum(r.completion_rate for r in results) / len(results),
            "avg_efficiency_ratio": sum(r.efficiency_ratio for r in results if r.completed) / max(sum(1 for r in results if r.completed), 1),
            "avg_waste_ratio": sum(r.waste_ratio for r in results) / len(results),
            "timestamp": datetime.now().isoformat(),
        }
        (output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))

        print(f"\n{'='*60}")
        print(f"RESULTS: {args.variant}")
        print(f"{'='*60}")
        print(f"Completed: {summary['completed']}/{summary['tasks_run']}")
        print(f"Avg completion: {summary['avg_completion_rate']:.1%}")
        print(f"Avg efficiency: {summary['avg_efficiency_ratio']:.1f}x optimal")
        print(f"Avg waste: {summary['avg_waste_ratio']:.1%}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
