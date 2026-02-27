# Auth System

## Goal
Build JWT-based authentication with refresh token rotation for the QSR ordering platform. Supports 140k+ concurrent users.

## Current Milestone
auth-middleware

## Status
Implemented refresh token rotation in auth middleware — tokens now rotate on each use with 7-day expiry.

## Next Steps
1. Add rate limiting to /auth/refresh endpoint
2. Implement token blacklist for forced logout
3. Add session expiry redirect with return URL
4. Wire up multi-tab token sync via BroadcastChannel
5. Error response standardization ({error: string, code: string})
6. Integration tests for full auth flow

## Active Context
- Using httpOnly cookies for token storage (XSS mitigation)
- Refresh tokens stored in Redis with user_id index
- ⚠️ Rate limiter is per-IP — revisit before launch for shared office IPs

## Completed Milestones

### Milestone: data-model (P-001 → P-008)
Designed and implemented user/session schema. Chose denormalized session table after P-005 schema rewrite — original was over-normalized for our query patterns. Key decision: store last_active as a column rather than computing from logs.

## Prompt History (Recent)
| ID | Summary |
|----|---------|
| P-042 | Added JWT refresh token rotation to auth middleware |
| P-041 | Debugged session expiry race condition — root cause was stale Redis cache |
| P-040 | Implemented validateToken() in auth/middleware.js |
| P-039 | Set up Redis connection pool for session store |
| P-038 | Designed refresh token rotation flow — chose sliding window over fixed expiry |
