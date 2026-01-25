# System Issue Report: Claude Model Quota Corruption (Critical)

**Reporting Date**: 2026-01-25
**Affected Models**: Claude 3.5 Sonnet, Claude 3.5 Thinking, Claude 3 Opus
**Scope**: Antigravity Pro Plan Users
**Version**: Observed post-v1.15.6

## 1. Description of the Bug

A severe regression in the quota management system is causing Pro users to be incorrectly subjected to "Weekly Reset" limits (approx. 120-135 hours) instead of the standard 5-hour reset.

## 2. Identified Symptoms

- **Reset Loop**: The usage reset timer extends itself upon reaching zero, preventing model recovery.
- **Immediate Lockout**: Models return "Resource exhausted" or "Code 429" even with 0% current usage.
- **Differential Failure**: Gemini models remain functional with 5-hour resets, while Anthropic/Claude models are exclusively partitioned into the failed weekly cooldown logic.

## 3. Evidence and Community Impact

Verified reports indicate a global issue affecting multiple Pro subscribers.

- [Google AI Developers Forum](https://discuss.ai.google.dev/t/bug-antigravity-ide-critical-quota-error-7-day-lockout-for-google-ai-pro-subscriber/114724)
- [Reddit r/google_antigravity](https://www.reddit.com/r/google_antigravity/comments/1qlnmaf/antigravity_quota_limit_should_have_resumed_today)

## 4. Requested Action (Backend Team)

- Investigate the quota fallback logic introduced in v1.15.6.
- Manually reset the `quota_limit_weekly` flags for Pro users affected by the loop.
- Sync reset timers with the 5-hour periodicity defined in the Pro Service Level Agreement (SLA).

---

_This report was automatically filed by the Antigravity Agent (Gemini) upon user request to preserve context and escalate platform blockers._
