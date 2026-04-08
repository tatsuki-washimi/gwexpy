# Pull Request Description

Fixes # (issue)

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context.

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional change)

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have checked my code with `ruff` and `mypy` locally
- [ ] I have updated the `CHANGELOG.md` if necessary

## Acceptance Criteria
- [ ] `cd docs && make html` (ja/en) で警告が出ないことを確認した
- [ ] `python scripts/check_terms.py` がパスした (用語揺れなし)
- [ ] `python scripts/check_docs_sync.py` がパスした (日・英の構造一致)
- [ ] `python scripts/check_external_links.py` を実行、またはリンク有効性を確認した
- [ ] Quickstart のコードブロックが正常に動作することを確認した
