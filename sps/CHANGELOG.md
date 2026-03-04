<!-- doc-id: f8965ead-c373-489b-868d-c97d7436a275 -->
# SPS Changelog

All notable changes to the Stateful Project System prompt.

Versioning: `MAJOR.MINOR.PATCH`
- **MAJOR** — fundamental restructuring, new core files, breaking changes to artifact format
- **MINOR** — new features, new config flags, new fields on existing artifacts
- **PATCH** — wording fixes, checklist tweaks, clarifications with no structural impact

---

## v4.3.0 — 2026-03-03

### Changed
- **`trail_archiving` upgraded to enum** — now accepts `off | github | db | both` (was boolean `off | on`)
- `github` → archives via `commit_trail` to build-logs repo (previous "on" behavior)
- `db` → stores via `log_build_trail` to Supabase, queryable via `query_build_trails`
- `both` → archives to both GitHub and Supabase
- Phase 5 Ship and Feature Transitions updated to reference both backends
- Config display shows valid values: `(off | github | db | both)`
- Config toggle supports enum syntax: `9db`, `9github`, `9both`, `9off`
- Infrastructure note updated to document both optional integrations

### Added
- **`query_build_trails` MCP tool** — retrieve stored trail artifacts from Supabase. Filter by project, type, milestone. Supports `latest_per_type` mode for restoring full trail state.

## v4.2.0 — 2026-03-03

### Added
- **SPS Lite mode** (`/lite`) — generates a stripped-down prompt with only user-selected features
- **Security-enhanced code review** — Phase 2 now checks for hardcoded secrets, .env exposure, tokens in localStorage, injection vectors, auth gaps, CORS misconfig, and other structural security anti-patterns
- **Interview presets** — `interview_mode` now accepts `off|on|build|fix|refactor|learn` with tailored question sets per preset
- **`/learn` mode** — exploratory mode for understanding codebases/concepts, no Phase 2-5 lifecycle
- **`trail_archiving` config flag** (off by default) — gates all commit_trail references; when off, trail files stay local
- **`sps_version` field** on HISTORY.jsonl entries — semver from prompt header, set once at session start
- **`sps_mode` field** on HISTORY.jsonl entries — tracks active mode (build/review/test/docs/retro/learn)
- **`context_tokens` field** on response entries (optional) — approximate context window usage for correlating quality with context pressure
- Compliance scorer checks for `sps_version`, `sps_mode`, and `context_tokens` fields

### Changed
- SPS_CONFIG expanded to 9 flags (was 8)
- Session start config display now shows `SPS v4.2.0:` with semver
- Retro template uses semver in `SPS Version` field
- Feature Transitions and Phase 5 Ship archive steps gated behind `trail_archiving` flag
- commit_trail now instructed to use descriptive messages: `"Archive [project]: [milestone] (P-XXX → P-YYY)"`

## v4.1.0 — 2026-03-03

### Added
- **`session_id`** field on every HISTORY.jsonl entry — format `s-YYYYMMDD-XXXX`, generated at SESSION START
- **CHANGELOG.md** as core SPS file — Keep Changelog format (Added/Changed/Fixed/Removed), appended at Phase 5 Ship
- **SPS Version and Sessions** in RETRO.md header — retro protocol collects session_ids and writes them
- **Infrastructure independence note** — documents zero hard dependencies on external services
- Compliance scorer checks for session_id presence, format, retro SPS version, and retro session list

### Changed
- SESSION START expanded to 10 steps (was 9) — step 1 generates session_id
- Config display shows session ID alongside config flags
- Retro protocol steps 3 and 7 handle session_id collection and writing

## v4.0.0 — 2026-03-03

### Added
- **SPS_CONFIG** — 8 feature flags with per-project override via PLAN.md
- **5-phase Feature Lifecycle** — Build → Code Review → Test Assembly → Documentation → Ship
- **Feature Transitions** — T-XXX entries, archive/reset protocol for multi-feature sessions
- **Code review** (Phase 2) — self-review protocol with 5-point checklist, `review` process tag
- **Test Assembly** (Phase 3) — write tests from TEST_QUEUE.md breadcrumbs, `test-assembly` tag
- **Documentation** (Phase 4) — write docs from `docs_worthy` entries, `docs-write` tag
- **Quality process tags** — `review`, `test-assembly`, `docs-write` (new category alongside productive/waste/neutral)
- **`docs_worthy` field** on response entries — flags architectural decisions for Phase 4
- **Test types** on TEST_QUEUE.md entries — `unit`, `e2e`, `manual` with type-specific templates
- **`/test`**, **`/docs`**, **`/review`**, **`/transition`** modes
- Compliance scorer: quality tags, transition validation, docs_worthy checks, 4-category process health, Feature Lifecycle checks

### Changed
- Process tag vocabulary expanded from 12 to 15 tags (3 quality tags added)
- Process Targets now track 4 categories (was 3): productive, quality, waste, neutral
- Every-turn protocol includes code review trigger check
- Retro template includes Code Review section

## v3.0.0 — 2026-02-27

Initial public release as part of ai-tools repo. Core SPS with PLAN.md, HISTORY.jsonl, TEST_QUEUE.md, RETRO.md, PLAYBOOK.md, process tagging, and retro protocol.

## v2.0.0 — 2026-02-25

Earlier version with simplified structure. Historical reference at `sps/agent-plan-prompt-v2.md`.