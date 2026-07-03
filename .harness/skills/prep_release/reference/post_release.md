# 公開後検証 (Post-Release Gates)

PyPI タグ公開後に順番に実施する段階検証。

## Step 1: PyPI 公開後 smoke テスト

クリーンな venv でインストールし、基本動作を確認。

```bash
python -m venv /tmp/gwexpy-smoke && source /tmp/gwexpy-smoke/bin/activate
pip install gwexpy==X.Y.Z
python -c "import gwexpy; gwexpy.register_all()"
deactivate && rm -rf /tmp/gwexpy-smoke
```

**GO/NO-GO**: `import` 失敗または `register_all()` が例外を送出した場合は **NO-GO**。
直ちにロールバック手順（`yank` → ホットフィックス）へ移行する。

## Step 2: GitHub Release 本文確認

1. GitHub Release ページ（`gh release view vX.Y.Z --web` で開ける）を確認する
2. `CHANGELOG.md` の同バージョンセクションと内容が一致しているか照合する

**GO/NO-GO**: 重大な差異（バージョン番号ミス・空 body）は **NO-GO**。
軽微な表記ゆれは次リリース前に修正で可。

## Step 3: Zenodo / メタデータ整合確認

```bash
python scripts/check_release_metadata.py
```

確認項目:
- `pyproject.toml` / `gwexpy/_version.py` / `CITATION.cff` / `.zenodo.json` のバージョン一致
- `CITATION.cff` の `date-released` が本日日付と一致

**GO/NO-GO**: バージョン不一致またはスクリプトが非ゼロ終了した場合は **NO-GO**。
修正後に `git commit --amend` せず新コミットで再タグを検討する。

## Step 4: conda-forge feedstock 更新確認

PyPI 公開後 1〜2 時間以内に `regro-cf-autotick-bot` が feedstock PR を自動作成する。

確認手順:
1. `https://github.com/conda-forge/gwexpy-feedstock/pulls` で PR の存在を確認
2. PR タイトルのバージョン番号が `vX.Y.Z` と一致していることを確認
3. CI が通過したら `merged` になるまで待機（マージは bot が行う）

**GO/NO-GO**: 24 時間経過後も PR が未作成の場合は conda-forge チームに手動報告。

## 失敗時の共通対応

| 状況 | 対応 |
|------|------|
| smoke テスト失敗 | PyPI で `yank`、`X.Y.Z+1` でホットフィックスリリース |
| メタデータ不整合 | 修正コミット → 新タグ（古いタグは削除可） |
| feedstock PR 未作成 | `conda-forge/gwexpy-feedstock` に手動 Issue |
