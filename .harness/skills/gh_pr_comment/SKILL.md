---
name: gh_pr_comment
description: "Post concise GitHub pull request comments with `gh pr comment`, especially when the GitHub connector cannot write comments or local review findings need to be attached to a PR."
---

# gh_pr_comment

## Goal
Attach verified local findings to a GitHub pull request using the GitHub CLI, with a connector-first fallback path and clear evidence.

## Trigger
Use this skill when:
- The user asks to comment on a PR.
- The GitHub connector cannot post because of token or permission errors.
- A local review/test result must be recorded on a PR.

## Required Inputs
- Repository owner/name, or a resolvable local git remote.
- Pull request number.
- Comment body containing concrete findings, verification commands, and unresolved risks.

## Workflow
1. Resolve the repository and PR target.
   - Prefer the local `origin` remote when the user does not specify a repo.
   - Confirm the PR exists with:
     ```bash
     gh pr view <PR_NUMBER> --repo <OWNER>/<REPO> --json number,title,state,url,headRefName,baseRefName
     ```
2. Prepare a concise Markdown comment.
   - Include exact commands run and outcomes.
   - Include file/function names for findings.
   - Avoid claiming checks passed when tests were skipped or not run.
3. Post with `gh pr comment`.
   ```bash
   gh pr comment <PR_NUMBER> --repo <OWNER>/<REPO> --body '<COMMENT_BODY>'
   ```
4. Report the returned comment URL to the user.

## Guardrails
- Do not post speculative findings. State uncertainty explicitly.
- Do not include secrets, local-only paths containing sensitive data, or raw tokens.
- If `gh` cannot connect in the sandbox, retry with escalated network permission.
- If `gh auth status` reports an invalid token but `gh pr view` works with escalation, continue using the working `gh pr` path.
- If posting fails, provide the exact comment body for manual posting.

## Output Template
```text
PR #<number> にコメント投稿しました。

<comment-url>
```

If posting fails:
```text
PR #<number> へのコメント投稿は失敗しました。

エラー:
<error>

投稿予定本文:
<body>
```
