# AI Model Selection Guide

*Quick reference for choosing the right model for Playwright work*

## Default Decision Tree

**Ask: Do I need to understand how things fit together?**
- YES → Opus (architecture, new features, complex helpers)
- NO → Sonnet (execution, repairs, individual tests)

**Ask: Is this almost mechanical?**
- YES → Haiku (probably won't use much)

---

## Common Tasks

### Opus territory:
- Initial feature exploration
- Table handlers (complex)
- Business logic test planning
- Coverage analysis + gap filling
- "Figure out how this works"

### Sonnet territory:
- Test repairs (fixing locators, event chains, timing)
- Element tests (discrete, pattern-based)
- Auth tests (unless super complex)
- Datepicker fixes/additions
- Executing from a plan
- "Build what I described"

### Switch Opus → Sonnet when:
- Plan is written in markdown
- Moving from exploration to execution
- You understand what needs to happen

### Escalate Sonnet → Opus when:
- Sonnet fails 2-3 times
- Needs architectural changes
- Solution works but feels wrong

---

## Your Upcoming Work

**Element tests review/refine**: Sonnet (batch similar elements)
**Auth tests**: Sonnet unless multi-role complexity
**Datepicker fixes**: Sonnet
**Table handlers**: Opus for architecture → Sonnet for implementation  
**Business logic tests**: Opus for planning → Sonnet for execution

**Note**: Most of your current work is Sonnet territory since locators are mapped. Save Opus for table handlers and business logic planning.

---

## Cost Reality Check

**Red flag**: $50+ overage = wrong model or scope creep
**Justified spend**: Foundation work, prevents rework, one-time architectural understanding
**Common waste**: Using Opus for repairs, running tests in AI instead of terminal

---

## Override Rule

Stuck and frustrated? Use Opus. Mental clarity > cost optimization.
