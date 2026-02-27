# Test Queue

## Ready to Write

### TQ-012: Session expiry redirect (Milestone: auth-middleware, P-041)
- **What:** JWT expires mid-session → redirect to /login with return URL
- **Happy path:** /dashboard → token expires → /login?return=/dashboard → re-auth → back
- **Error states:** Expired refresh token (show message, no loop). Network fail during refresh.
- **Edge cases:** Token expires during form submit. Multi-tab sync.
- **Selectors:** `[data-testid="login-form"]`, `[data-testid="session-expired-banner"]`
- **Setup:** Requires expired token state — `utils.setExpiredAuthState()` or clock mock

### TQ-013: Refresh token rotation (Milestone: auth-middleware, P-042)
- **What:** On token refresh, old refresh token invalidated, new one issued
- **Happy path:** Request with expired access token → auto-refresh → new tokens → original request succeeds
- **Error states:** Reuse of old refresh token (should fail). Concurrent refresh requests.
- **Selectors:** N/A (API-level test)
- **Setup:** Requires valid refresh token in Redis

## Deferred

### TQ-008: Rate limiting feedback (Milestone: api-layer, P-028)
- **Blocked by:** Rate limit headers not yet implemented server-side
- **When ready:** After rate limit endpoint lands
