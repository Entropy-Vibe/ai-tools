# Context Manifest — Token Budget & Prompt Inventory Tool

A CLI tool for prompt engineers who treat context like a measurable resource budget.

## Why

When building agent orchestrators, eval frameworks, or iterating on prompt assemblies, you need to know:
- **What's in my context?** — inventory of all files feeding into a prompt
- **How much budget am I using?** — token counts per file and total
- **What changed?** — diff between runs to track additions/removals
- **Is this worth it?** — log runs with tags to correlate context composition with eval results

## Quick Start

```bash
# Scan current directory
python context_manifest.py .

# Scan specific files
python context_manifest.py plan.md agents.md pw_agents.md mentor.md

# Scan with a budget and tag for eval tracking
python context_manifest.py ./prompts --budget 200000 --tag "baseline_v2" --log runs.csv

# Watch for changes (re-scans every 2s)
python context_manifest.py ./prompts --watch
```

## Output Formats

### Table (default)
```
╔══════════════════════════════════════════════════════════════╗
║  CONTEXT MANIFEST  [baseline_v2]                            ║
╠════╦══════════════════════════════╦════════╦═══════╦════════╣
║  # ║ File                         ║ Tokens ║ Lines ║   KB   ║
╠════╬══════════════════════════════╬════════╬═══════╬════════╣
║  1 ║ pw_agents.md                 ║   2.3k ║   180 ║   9.2  ║
║  2 ║ mentor.md                    ║   1.8k ║   142 ║   7.1  ║
║  3 ║ plan.md                      ║   0.9k ║    65 ║   3.4  ║
╠════╩══════════════════════════════╬════════╩═══════╩════════╣
║  Total Context                    ║    5.0k tokens          ║
║  Budget (200.0k)                  ║  195.0k remaining       ║
║  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2.5%                    ║
╚══════════════════════════════════════════════════════════════╝
```

### JSON (`--format json`)
Machine-readable output for piping into your eval framework.

### Markdown (`--format markdown`)
Paste into docs or PR descriptions.

### Compact (`--format compact`)
Minimal one-line-per-file for terminal workflows.

## Key Features

### Eval Correlation (`--tag` + `--log`)
Tag each run and log to CSV. Over time, correlate which context compositions produce better eval results:
```bash
python context_manifest.py ./prompts --tag "with_rag_v3" --log eval_runs.csv
python context_manifest.py ./prompts --tag "no_rag_baseline" --log eval_runs.csv
```

### Diff Mode (`--diff`)
Compare current context against a saved snapshot:
```bash
# Save a baseline
python context_manifest.py ./prompts --save baseline.json

# Later, see what changed
python context_manifest.py ./prompts --diff baseline.json
```

### Watch Mode (`--watch`)
Live-updating display as you edit files — great for iterating on prompts.

### Sorting (`--sort`)
Sort by `tokens` (default), `name`, `lines`, or `size`.

## Token Estimation

Uses heuristic analysis (~5-10% of real counts) that accounts for:
- Prose vs code content density
- URLs and file paths (tokenize into many small tokens)
- Markdown formatting overhead

Good enough for budgeting. For exact counts, pipe the JSON output into the Anthropic token counting API.

## Zero Dependencies

Pure Python 3.10+. No external packages required.
