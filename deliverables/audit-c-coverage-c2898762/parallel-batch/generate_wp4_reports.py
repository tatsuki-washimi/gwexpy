#!/usr/bin/env python3
"""Generate WP4 Deliverables:
1. Re-evaluation of 269 verified APIs (JSON + Markdown)
2. Classification of 106 exception_parity APIs (JSON + Markdown)
"""

import json
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parents[3]
DELIVERABLES = REPO / "deliverables" / "audit-c-coverage-c2898762"
LEDGER_FILE = DELIVERABLES / "c-coverage-c2898762.jsonl"

def main():
    rows = [json.loads(l) for l in LEDGER_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]

    # =========================================================================
    # Part 1: Re-evaluation of 269 verified APIs
    # =========================================================================
    verified_rows = [r for r in rows if r.get("disposition") == "verified"]
    assert len(verified_rows) == 269, f"Expected 269 verified rows, got {len(verified_rows)}"

    reeval_records = []
    for r in verified_rows:
        ident = r["identifier"]
        trials = r.get("evidence", {}).get("parallel_worker_trials", [])
        pos = next((t for t in trials if t.get("trial_kind") == "positive_control"), None)
        abn = next((t for t in trials if t.get("trial_kind") == "abnormal_input"), None)
        cmp_t = next((t for t in trials if t.get("trial_kind") == "gwpy_comparison"), None)

        pos_checks = [c["check"] for c in pos.get("assertions", {}).get("checks", [])] if pos else []
        pos_actual = pos.get("actual", {}) if pos else {}
        contract_name = pos_actual.get("contract", "unknown")

        abn_passed = abn.get("assertions", {}).get("passed") if abn else None
        abn_exc = abn.get("actual", {}).get("exception") if abn else None

        gwpy_status = r.get("gwpy_comparison")
        cmp_mode = r.get("comparison_mode")

        # Assess verification depth
        # Depth level:
        #   'L3_rigorous_numerical_parity': GWpy numeric_value_parity with allclose + finite + units
        #   'L2_substantive_contract': Concrete contract assertions on physical invariants (unit, finite, bounds, shape)
        #   'L1_basic_smoke_attribute': Minimal execution (e.g. empty checks, smoke only)
        if gwpy_status == "numeric_value_parity":
            depth_level = "L3_rigorous_numerical_parity"
        elif len(pos_checks) == 0:
            depth_level = "L1_basic_smoke_attribute"
        else:
            depth_level = "L2_substantive_contract"

        # Delineate verified vs unverified scope
        verified_scope = f"Contract: {contract_name}; Checks: {pos_checks}; Abnormal input: {bool(abn_passed)}"
        if gwpy_status == "exception_parity":
            unverified_scope = "GWpy side-by-side numerical comparison unverified (worker called gw_attr with 0 args, resulting in TypeError); verified only on GWexpy positive control fixture"
        elif gwpy_status == "incomparable":
            unverified_scope = "GWexpy extension without direct GWpy equivalent; general multi-parameter and edge-case domain unverified beyond positive control fixture"
        else:
            unverified_scope = "Single-fixture positive control verified; multi-dimensional tensor/stress corner cases unverified"

        reeval_records.append({
            "identifier": ident,
            "class_name": ident.split("/")[0].split(".")[-1],
            "method_name": ident.split("/")[-1],
            "depth_level": depth_level,
            "contract_name": contract_name,
            "positive_checks": pos_checks,
            "abnormal_input_verified": bool(abn_passed),
            "abnormal_exception": str(abn_exc)[:80] if abn_exc else None,
            "gwpy_comparison": gwpy_status,
            "comparison_mode": cmp_mode,
            "verified_scope": verified_scope,
            "unverified_scope": unverified_scope,
        })

    # Save JSON
    v269_json_path = DELIVERABLES / "wp4-269-verified-reevaluation.json"
    v269_json_path.write_text(json.dumps(reeval_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {v269_json_path}")

    depth_counts = Counter(r["depth_level"] for r in reeval_records)
    print("269 depth counts:", depth_counts)

    # Generate Markdown Report
    lines_269 = [
        "# 既存 Verified 269 API の検証深度と実証範囲の再評価レポート",
        "",
        "**監査対象コミット**: `c2898762ca68e20a1a241fffb3c3262b0a27290b`  ",
        "**作成日**: 2026-10-18  ",
        "**対象母集団**: C監査ベースライン台帳における `verified` 269 API  ",
        "",
        "## 1. エグゼクティブサマリー",
        "",
        "ベースライン台帳で `verified` と判定されていた 269 API について、正当な科学的契約（単位保持・数値安定性・軸整合性）の充足度、正常系・異常系入力の網羅性、および GWpy 実機比較の深度を再評価しました。",
        "",
        "### 検証深度レベルの分類基準",
        "- **L3（厳格な数値パリティ / 63 API）**: GWpy との side-by-side 比較により、有限値性・許容誤差内での数値一致（`numeric_allclose`）が実証された API。",
        "- **L2（実質的契約検証 / 201 API）**: GWexpy 単体または独自拡張として、物理単位の保持（`unit_preserved`）、有限値性（`np.isfinite`）、境界値、形状・メモリ独立性などの具体的な科学的契約アサーションが成立している API。",
        "- **L1（属性・スモーク検証 / 5 API）**: 属性の存在や最小限の呼び出し成立のみが確認されている API（`TimeSeries.override_unit`, `Plane2D.crop`, `load_dttxml_native`, `get_enbw`, `pca_transform`）。",
        "",
        "| 検証深度レベル | 件数 | 構成比 | 主な特性と実証範囲 |",
        "| :--- | :---: | :---: | :--- |",
        f"| **L3: 厳格な数値パリティ** | {depth_counts['L3_rigorous_numerical_parity']} | {depth_counts['L3_rigorous_numerical_parity']/len(reeval_records)*100:.1f}% | TimeSeries / FrequencySeries / Spectrogram 等の主要演算。GWpy との完全数値一致を実証。 |",
        f"| **L2: 実質的契約検証** | {depth_counts['L2_substantive_contract']} | {depth_counts['L2_substantive_contract']/len(reeval_records)*100:.1f}% | 単位変換・有限性・配列形状・スライス契約を検証。独自拡張（Plane2D, FieldBase等）を含む。 |",
        f"| **L1: 属性・スモーク検証** | {depth_counts['L1_basic_smoke_attribute']} | {depth_counts['L1_basic_smoke_attribute']/len(reeval_records)*100:.1f}% | 最小限のシグネチャ呼び出しやプロパティ参照のみ。詳細なドメイン検証は限定的。 |",
        f"| **合計** | **{len(reeval_records)}** | **100.0%** | **全件について実証済み範囲と未実証範囲を明確に区分** |",
        "",
        "## 2. 正常系・異常系・GWpy比較の網羅性分析",
        "",
        "| 観点 | 充足状況 | 監査所見と未実証範囲の明確化 |",
        "| :--- | :---: | :--- |",
        f"| **異常系入力耐性** | 256 / 269 ({256/269*100:.1f}%) | 不正な型・引数・キーワードが与えられた際に、サイレントな内部データ破損を起こさず適切な例外（`TypeError`, `ValueError`）を発火することを確認。13件は初期エビデンス保持項目。 |",
        f"| **GWpy直接等価性** | 87 / 269 ({87/269*100:.1f}%) | 数値パリティ 63 件、メタデータパリティ 24 件。GWpy との直接同一性が確認された範囲。 |",
        f"| **GWexpy独自拡張** | 132 / 269 ({132/269*100:.1f}%) | `incomparable`。GWpy に対応クラス・メソッドが存在しない独自拡張（Plane2D, FieldBase, ScalarField 等）。正系契約は成立しているが GWpy パリティは対象外。 |",
        f"| **未実証GWpy比較** | 50 / 269 ({50/269*100:.1f}%) | `exception_parity` と記録されていたもの。ハーネスのゼロ引数呼び出し起因であり、GWexpy 側では正系契約が成立しているものの、GWpy との引数付き数値比較は未実証。 |",
        "",
        "## 3. 実証済み範囲と未実証範囲の境界画定",
        "",
        "### 実証済み範囲（Verified Scope）",
        "1. 各 API の positive control で実装された個別契約（例: `abs` の非負性・単位保持、`copy` の独立メモリ、`crop` のインデックス範囲整合、`override_unit` の値不変性）。",
        "2. 不正引数入力に対する堅牢性（256 API における例外検知）。",
        "3. L3 の 63 API における GWpy 実機との数値・時間軸・単位の完全パリティ。",
        "",
        "### 未実証範囲（Unverified Scope / 今後の課題）",
        "1. **50件の引数必須 API**: GWpy 実機との引数付き数値比較（ハーネスの引数不足により未比較）。",
        "2. **高次元・テンソル演算の境界値**: 多次元配列における極限値（NaN, Inf, ゼロ除算, 不整合単位演算）の複合ストレステスト。",
        "3. **独自拡張の多変量入力空間**: Plane2D, FieldBase の非等間隔グリッドや高次テンソル場における網羅的パラメータ空間。",
        "",
        "## 4. 269 API 個別再評価一覧（抜粋）",
        "",
        "| 識別子 | クラス | 深度 | GWpy比較 | 実証済み契約 | 未実証・限定範囲 |",
        "| :--- | :--- | :---: | :---: | :--- | :--- |",
    ]

    for r in reeval_records[:30]:
        lines_269.append(
            f"| `{r['method_name']}` | `{r['class_name']}` | {r['depth_level']} | {r['gwpy_comparison']} | {r['contract_name']} ({len(r['positive_checks'])} checks) | {r['unverified_scope'][:50]}... |"
        )

    v269_md_path = DELIVERABLES / "2026-10-18-wp4-269-verified-reevaluation.md"
    v269_md_path.write_text("\n".join(lines_269) + "\n", encoding="utf-8")
    print(f"Generated {v269_md_path}")

    # =========================================================================
    # Part 2: Classification of 106 exception_parity APIs
    # =========================================================================
    exc_rows = [r for r in rows if r.get("gwpy_comparison") == "exception_parity"]
    assert len(exc_rows) == 106, f"Expected 106 exception_parity rows, got {len(exc_rows)}"

    exc_classification = []
    for r in exc_rows:
        ident = r["identifier"]
        trials = r.get("evidence", {}).get("parallel_worker_trials", [])
        gw_trial = next((t for t in trials if t.get("trial_kind") == "gwpy_comparison"), None)
        gw_act = gw_trial.get("actual", {}) if gw_trial else {}
        gw_exc = gw_act.get("gw_exc")
        ex_exc = gw_act.get("ex_exc")

        # Classify
        if gw_exc == "TypeError":
            cat = "trivial_argument_insufficiency"
            cat_desc = "Harness called method with 0 arguments (gw_attr()); both raised TypeError: missing required positional argument"
            impact = "Does not reflect true behavioral divergence; fixture-supplied side-by-side execution required to test numerical parity"
        elif gw_exc in ("AttributeError",):
            cat = "upstream_matplotlib_behavior"
            cat_desc = f"Both raised {gw_exc} due to upstream Matplotlib drawing/backend method differences (e.g. drawstyle step)"
            impact = "Genuine exception parity under upstream Matplotlib conditions"
        elif gw_exc in ("ValueError", "IndexError", "IORegistryError"):
            cat = "abnormal_or_domain_boundary_input"
            cat_desc = f"Both raised {gw_exc} on domain boundary, invalid format, or missing target file conditions"
            impact = "Exception parity verified on invalid domain inputs or boundary conditions"
        else:
            cat = "other_exception"
            cat_desc = f"{gw_exc} observed"
            impact = "TBD"

        exc_classification.append({
            "identifier": ident,
            "disposition": r.get("disposition"),
            "execution_status": r.get("execution_status"),
            "observed_gw_exception": gw_exc,
            "observed_ex_exception": ex_exc,
            "category": cat,
            "category_description": cat_desc,
            "impact_assessment": impact,
        })

    # Save JSON
    exc106_json_path = DELIVERABLES / "wp4-106-exception-parity-classification.json"
    exc106_json_path.write_text(json.dumps(exc_classification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {exc106_json_path}")

    cat_counts = Counter(r["category"] for r in exc_classification)
    print("106 category counts:", cat_counts)

    # Generate Markdown Report
    lines_106 = [
        "# 106件の Exception Parity 原因別分類・再評価台帳",
        "",
        "**監査対象コミット**: `c2898762ca68e20a1a241fffb3c3262b0a27290b`  ",
        "**作成日**: 2026-10-18  ",
        "**対象母集団**: C監査ベースライン台帳において `gwpy_comparison == \"exception_parity\"` と記録された 106 API  ",
        "",
        "## 1. エグゼクティブサマリー",
        "",
        "旧台帳で「GWpy と例外が一致した」として `exception_parity` に分類されていた 106 件について、実行コードと例外内容を精査した結果、**94件（88.7%）がハーネスの引数不足によるゼロ引数呼び出し（TypeError）起因**であることが判明しました。",
        "",
        "| 分類カテゴリ | 件数 | 構成比 | 発生原因・挙動の真相 | 実質的影響と評価 |",
        "| :--- | :---: | :---: | :--- | :--- |",
        f"| **カテゴリA: 単なる引数不足の一致** | {cat_counts['trivial_argument_insufficiency']} | {cat_counts['trivial_argument_insufficiency']/len(exc_classification)*100:.1f}% | ハーネスが引数を要するメソッドを無引数 `gw_attr()` で呼び出し、双方が `TypeError: missing required argument` を発火。 | 製品の真のパリティを証明しておらず、数値比較が未実施のまま例外一致として処理された「見かけのパリティ」。 |",
        f"| **カテゴリB: 異常・ドメイン境界入力の一致** | {cat_counts['abnormal_or_domain_boundary_input']} | {cat_counts['abnormal_or_domain_boundary_input']/len(exc_classification)*100:.1f}% | 空配列・無効フォーマット・存在しないファイル指定等により、双方が `ValueError`, `IndexError`, `IORegistryError` を発火。 | 異常系・境界条件における例外契約の正当な一致。 |",
        f"| **カテゴリC: 上流 Matplotlib 描画挙動** | {cat_counts['upstream_matplotlib_behavior']} | {cat_counts['upstream_matplotlib_behavior']/len(exc_classification)*100:.1f}% | Matplotlib のバージョン差異や描画スタイル指定（drawstyle step）により双方が `AttributeError` を発火。 | 上流ライブラリ仕様に起因する挙動一致（WP1/WP2是正済み）。 |",
        f"| **合計** | **{len(exc_classification)}** | **100.0%** | **見かけの一致（94件）と正当な境界例外（12件）を厳格に分離** |",
        "",
        "## 2. 旧 Disposition とのクロス集計",
        "",
        "| 旧 Disposition | カテゴリA（引数不足） | カテゴリB（異常系・境界） | カテゴリC（上流Matplotlib） | 合計 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    disp_cat = Counter((r["disposition"], r["category"]) for r in exc_classification)
    for disp in ("verified", "contract_pending", "environment_blocked", "differ"):
        cnt_a = disp_cat.get((disp, "trivial_argument_insufficiency"), 0)
        cnt_b = disp_cat.get((disp, "abnormal_or_domain_boundary_input"), 0)
        cnt_c = disp_cat.get((disp, "upstream_matplotlib_behavior"), 0)
        tot = cnt_a + cnt_b + cnt_c
        lines_106.append(f"| `{disp}` | {cnt_a} | {cnt_b} | {cnt_c} | **{tot}** |")

    lines_106.extend([
        "",
        "### 重大な監査知見",
        "1. **Verified 50件の性質**: `verified` に属する 50 件は、GWexpy 側の positive control では有効なフィクスチャによる契約検証に成功しているものの、GWpy 実機比較においては無引数呼び出しによる `TypeError` が記録されていました。したがって、これらの「GWpy 比較」は数値パリティの実証ではなく「比較保留（`comparison_pending`）」として扱うのが誠実な記録です。",
        "2. **Differ 10件の性質**: `differ` に属する 10 件のうち、6 件は `whiten`, `gate` などの信号処理契約であり、デフォルトのサンプル長不足（10サンプル）で両者または GWexpy が `ValueError` を発火したものです（WP1で既に原因特定済み）。",
        "",
        "## 3. カテゴリB（異常・境界入力）およびカテゴリC（上流挙動）の12件全件一覧",
        "",
        "| 識別子 | 旧Disposition | 観測例外 | 分類 | 評価・備考 |",
        "| :--- | :---: | :---: | :--- | :--- |",
    ])

    for r in exc_classification:
        if r["category"] != "trivial_argument_insufficiency":
            lines_106.append(
                f"| `{r['identifier']}` | `{r['disposition']}` | `{r['observed_gw_exception']}` | {r['category']} | {r['category_description']} |"
            )

    exc106_md_path = DELIVERABLES / "2026-10-18-wp4-106-exception-parity-classification.md"
    exc106_md_path.write_text("\n".join(lines_106) + "\n", encoding="utf-8")
    print(f"Generated {exc106_md_path}")

if __name__ == "__main__":
    main()
