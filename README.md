# ai-tools

Public collection of AI development tools and frameworks.

## Contents

### [sps/](sps/) — Stateful Project Steward
System prompt that turns Claude into a stateful coding agent with persistent project state (PLAN.md, HISTORY.jsonl), self-tagging process awareness, test breadcrumbing, and structured retrospectives.

### [eval-framework/](eval-framework/) — Eval Framework for SPS
A/B testing and evaluation framework for the SPS prompt. Four tools that evaluate both sides of the human-agent collaboration loop — agent compliance and output quality, and human prompting patterns.

- **Compliance Scorer** — Structural validation (free, local)
- **Task Bank + Runner** — 20 synthetic tasks for controlled A/B testing
- **Judge Pipeline** — AI-powered output quality evaluation
- **Prompt Analyzer** — Human prompting pattern analysis
- **Session Replay** — Retro validation via optimal path replay

See [eval-framework/README.md](eval-framework/README.md) for full documentation and cost breakdown.

### model-selection-guide.md
Historical reference for AI model selection by task type.

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```
