# Agent Plan Prompt v2 — Eval Framework

## What This Is

An evaluation and A/B testing framework for a system prompt that turns Claude into a stateful coding agent. The system prompt (`agent-plan-prompt-v2.md`) gives the agent persistent project state (PLAN.md, HISTORY.jsonl), self-tagging process awareness, test breadcrumbing, and structured retrospectives.

This framework answers: **does the system prompt actually make the agent better, and how do I get better at using it?**

It evaluates both sides of the collaboration loop — the agent's compliance and output quality, and the human's prompting patterns and habits.

## The System Prompt

The prompt defines a stateful project system with five core files:

**PLAN.md** — Living plan with goal, milestones, status, next steps, active context, and prompt history. Updated every turn. This is the agent's primary navigation.

**HISTORY.jsonl** — Append-only archive of every prompt/response pair with tags, summaries, and process metadata. Searchable by milestone and tags.

**TEST_QUEUE.md** — Test breadcrumbs captured at build time (selectors, setup requirements, happy/error/edge cases) so tests can be written later without re-discovering everything.

**RETRO.md** — After-action reviews at milestone boundaries. Tallies process tags, analyzes waste, reconstructs the optimal path, and proposes playbook additions.

**PLAYBOOK.md** — Cross-project learnings. Accumulated prompt patterns, anti-patterns, sequencing rules, and process targets. Seeded into new projects.

Every turn, the agent self-tags its response with a process tag from a fixed vocabulary (clean-execution, dead-end, redo, vague-prompt, misfire, etc.) and displays a Done/Next/Process footer so the human always knows where they are.

## The Eval Framework

```
eval-framework/
├── compliance_scorer.py           # 1. Structural validation
├── tasks/
│   ├── task_bank.json             # 2. 20 synthetic tasks
│   └── runner.py                  #    Multi-variant execution engine
├── judges/
│   └── judge_pipeline.py          # 3. AI judge for agent quality
├── analyzers/
│   └── prompt_analyzer.py         # 3.5. Human prompt pattern analysis
├── replay/
│   └── replay.py                  # 4. Retro validation via replay
└── fixtures/sample-session/       # Test data
```

### 1. Compliance Scorer

Validates that session artifacts conform to all structural contracts in the system prompt. Runs locally, no API key needed.

39 checks across 7 categories: PLAN.md sections and content quality, HISTORY.jsonl field validation and JSON integrity, process tag vocabulary and pairing, waste-tag process notes, TEST_QUEUE.md breadcrumb completeness, RETRO.md structure, cross-file ID and milestone sync, and process health metrics (productive/waste ratios, consecutive waste detection, tag honesty heuristics).

```bash
python compliance_scorer.py --session-dir ./my-project --verbose
python compliance_scorer.py --session-dir ./my-project --json > report.json
# Exit code 0 if score >= 80
```

### 2. Synthetic Task Bank + Runner

20 coding tasks designed for controlled A/B testing across 4 complexity tiers:

| Tier | Tasks | Optimal Turns | Purpose |
|------|-------|---------------|---------|
| Simple | 5 | 3-5 | Baseline overhead measurement |
| Medium | 5 | 7-9 | Process tag and state persistence value |
| Complex | 5 | 15-22 | Full system stress test |
| Adversarial | 5 | 2-10 | Specific failure mode exposure |

Adversarial tasks include: ambiguous requirements (tests whether agent scopes before building), cascading bugs (second bug masks third), premature optimization trap, mid-stream requirements pivot, and session recovery from half-completed state.

The runner executes tasks through variant system prompts and measures completion rate, turn efficiency, and waste ratio.

```bash
python tasks/runner.py list
python tasks/runner.py run --variant full --tier simple --output results/
python tasks/runner.py run --variant minimal --task-id T-006 --output results/
python tasks/runner.py compare results/full/ results/minimal/ results/none/
```

Variants: `full` (complete v2 prompt), `minimal` (PLAN + HISTORY only, no process tags), `none` (raw Claude), `custom` (your own prompt file).

### 3. Judge Pipeline

Uses a separate Claude instance to evaluate agent output quality per turn across three dimensions:

**Tag accuracy** — Pairs each prompt/response, asks the judge "is this process tag correct?" Computes agreement rate and a honesty score that penalizes flattering bias (agent tagging productive more than the judge agrees).

**Summary specificity** — Evaluates each HISTORY.jsonl summary against the prompt's rules: does it name files/functions, include the why, capture decisions? Grades excellent through vague.

**Breadcrumb completeness** — Evaluates each TEST_QUEUE.md entry for the 7 required/desirable subfields. Grades ready_to_write through unusable.

```bash
python judges/judge_pipeline.py --session-dir ./my-project
python judges/judge_pipeline.py --history HISTORY.jsonl --test-queue TEST_QUEUE.md
```

### 3.5. Prompt Quality Analyzer

Evaluates the human's prompting patterns. Runs in three modes:

**Live check** — Score a prompt before sending it. Catches vague verbs, kitchen-sink multi-concern prompts, premature optimization language, missing output specifications, detail spirals (stream-of-consciousness corrections), and assumed context. Works instantly with local regex analysis, optionally enhanced with an AI call.

```bash
python analyzers/prompt_analyzer.py check "Make the auth work" --no-ai
# Score: 27/100  Grade: F
# Antipatterns: assumed_context

python analyzers/prompt_analyzer.py check "Implement JWT middleware in auth/middleware.js: validate token from Authorization header, check expiry, return 401 with {error: 'token_expired'}" --no-ai
# Score: 70/100  Grade: C
```

**Session analysis** — Post-hoc analysis of all prompts in a HISTORY.jsonl. Scores each prompt individually, then detects multi-turn sequence patterns: vague→correct cycles (you gave specifics as corrections that should have been in the first prompt), scope creep spirals (2+ consecutive off-plan turns), debug loops (3+ turns on the same root cause), and ping-pong (alternating planning and building without committing). Builds a habit profile with ratings for specificity, scope control, and output specification.

```bash
python analyzers/prompt_analyzer.py session --session-dir ./my-project
python analyzers/prompt_analyzer.py session --session-dir ./my-project --no-ai  # local only
```

**Cross-session habits** — Analyzes prompting patterns across multiple projects. Tracks whether your prompting is improving over time and surfaces your most persistent antipatterns.

```bash
python analyzers/prompt_analyzer.py habits --dirs proj1/ proj2/ proj3/
```

### 4. Session Replay

Validates whether retro insights are actionable by replaying the optimal path and comparing results.

**Auto pipeline** — Reads RETRO.md, uses AI to extract the optimal prompt sequence (eliminating waste turns, merging vague→correct pairs), replays it through the agent, and compares original vs. replay on turns, waste ratio, and completion. Validates which retro insights actually prevented waste.

**A/B replay** — Same optimal path through different system prompt variants to isolate prompt-level impact vs. system-prompt-level impact.

```bash
# Full auto: extract → replay → compare
python replay/replay.py auto --session-dir ./my-project --output replay-results/

# Extract optimal path for review before replaying
python replay/replay.py extract --session-dir ./my-project --output optimal.jsonl

# Hand-written optimal path
python replay/replay.py manual --playbook optimal.jsonl --system-prompt prompt-v2.md

# A/B: same path through full vs minimal vs none
python replay/replay.py ab --playbook optimal.jsonl --variants full,minimal,none

# Compare existing results
python replay/replay.py compare --original ./my-project --replay replay-results/
```

## Cost Breakdown

All tools that call the API accept a `--model` flag. Default is Sonnet. Recommendations per tool are marked with ★.

### Per-Model Pricing (as of Feb 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Haiku 4.5 | $0.80 | $4.00 |
| Sonnet 4.5 | $3.00 | $15.00 |
| Opus 4.5 | $15.00 | $75.00 |

### Step 1: Compliance Scorer

**$0.** Runs locally, no API calls.

### Step 2: Task Runner (20 tasks, per variant)

Each task runs a multi-turn agent conversation plus one completion-evaluation judge call.

| Tier | Tasks | Avg Turns | Haiku | ★ Sonnet | Opus |
|------|-------|-----------|-------|----------|------|
| Simple | 5 | ~8 | ~$0.15 | ~$0.75 | ~$3.75 |
| Medium | 5 | ~20 | ~$0.50 | ~$2.50 | ~$12.50 |
| Complex | 5 | ~45 | ~$1.35 | ~$6.75 | ~$33.75 |
| Adversarial | 5 | ~15 | ~$0.30 | ~$1.50 | ~$7.50 |
| **Total (1 variant)** | 20 | | **~$2.30** | **~$11.50** | **~$57.50** |
| **Total (3 variants)** | 60 | | **~$7** | **~$35** | **~$173** |

★ **Recommendation:** Use Sonnet for the agent conversations. The task runner needs to produce realistic agent behavior — Haiku may not follow the system prompt's complex multi-file protocol reliably enough to produce meaningful comparisons.

**Budget-conscious start:** Run just the 5 simple tasks across 3 variants first (~$2.25 on Sonnet) to validate the framework before scaling up. The complex tier alone is half the total cost.

### Step 3: Judge Pipeline (per session)

One API call per HISTORY.jsonl entry for tag + summary judging, one per TEST_QUEUE entry for breadcrumb judging. A typical 30-40 turn session:

| Component | Calls | ★ Haiku | Sonnet | Opus |
|-----------|-------|---------|--------|------|
| Tag judging | ~35 | ~$0.10 | ~$0.50 | ~$2.50 |
| Summary judging | ~35 | ~$0.05 | ~$0.25 | ~$1.25 |
| Breadcrumb judging | ~5 | ~$0.01 | ~$0.05 | ~$0.25 |
| **Total per session** | | **~$0.16** | **~$0.80** | **~$4.00** |

★ **Recommendation:** Use Haiku. These are classification and pattern-matching tasks — Haiku handles them well at 1/5 the cost. Save Sonnet for cases where you disagree with Haiku's tag judgments and want a second opinion.

### Step 3.5: Prompt Quality Analyzer

| Mode | Haiku | Sonnet | Opus |
|------|-------|--------|------|
| `check` (local, `--no-ai`) | **$0** | **$0** | **$0** |
| `check` (AI-enhanced) | ~$0.01 | ~$0.03 | ~$0.15 |
| `session` (local, `--no-ai`) | **$0** | **$0** | **$0** |
| `session` (AI, 30 prompts) | ~$0.07 | ~$0.35 | ~$1.75 |
| `habits` (always local) | **$0** | **$0** | **$0** |

★ **Recommendation:** Use `--no-ai` by default — the local regex analyzer catches the big antipatterns for free. Use Haiku AI-enhancement when you want rewrites and nuanced scoring on a specific session. The `habits` command is always free since it needs speed across many sessions.

### Step 4: Session Replay (per session)

| Component | ★ Sonnet | Haiku | Opus |
|-----------|----------|-------|------|
| Extract optimal path | ~$0.03 | ~$0.01 | ~$0.15 |
| Replay (15-turn optimal) | ~$0.75 | ~$0.15 | ~$3.75 |
| Insight validation | ~$0.02 | ~$0.01 | ~$0.10 |
| **Total (`auto`)** | **~$0.80** | **~$0.17** | **~$4.00** |
| **Total (`ab`, 3 variants)** | **~$2.50** | **~$0.50** | **~$12.00** |

★ **Recommendation:** Use Sonnet for replay conversations (same reasoning as task runner — needs realistic agent behavior). The extraction and validation calls are cheap enough that model choice doesn't matter.

### Total Cost Summary

| Scenario | ★ Recommended | All-Haiku | All-Sonnet | All-Opus |
|----------|---------------|-----------|------------|----------|
| Quick validation (5 simple tasks, 1 variant, 1 session judged + analyzed + replayed) | **~$3** | ~$0.75 | ~$4 | ~$20 |
| Full A/B (20 tasks, 3 variants, 5 sessions judged + analyzed, 1 replay) | **~$40** | ~$9 | ~$50 | ~$250 |
| Comprehensive (20 tasks, 3 variants, all sessions judged, 5 replays, habit analysis) | **~$55** | ~$13 | ~$65 | ~$330 |

★ **Recommended mix:** Sonnet for agent conversations (task runner + replay) where behavioral fidelity matters. Haiku for judge calls and prompt analysis where it's classification work. Local mode for compliance scoring and prompt habit tracking. This gives you the best signal at roughly 60-70% of all-Sonnet cost.

## North Star Metric

**Optimal Path Ratio** from RETRO.md: `estimated_optimal_turns / actual_turns`

Track this across projects. A better system prompt and better prompting habits should push it closer to 1.0 over time. The replay tool (Step 4) validates whether your retros produce ratios that hold up when actually replayed.

## Setup

```bash
pip install anthropic --break-system-packages
export ANTHROPIC_API_KEY=sk-ant-...
```

## Recommended Workflow

```
1. Build a feature using agent-plan-prompt-v2
2. Run /retro at milestone boundary
3. python compliance_scorer.py --session-dir .                    # free
4. python judges/judge_pipeline.py --session-dir . --model haiku  # ~$0.16
5. python analyzers/prompt_analyzer.py session --session-dir . --no-ai  # free
6. python replay/replay.py auto --session-dir . --output replay/  # ~$0.80
7. python analyzers/prompt_analyzer.py habits --dirs proj1/ proj2/ # free
```

Total per-session eval cost with recommended models: **~$1**.
