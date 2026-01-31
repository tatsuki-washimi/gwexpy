# Timeseries A2 Inventory Diff Report

**対象 Commit Hash:** `728ff79cb81c393402eef55ec90d051cabca9d2f`

**実行方法:**
```bash
python -m tools.a2_inventory_check_timeseries \
    --csv tests/timeseries_all_defs_classified.csv \
    --package gwexpy/timeseries
```

---

## 1. 台帳にあるが実コードに存在しない

| qualified_name | file | lineno | 推定原因 |
|----------------|------|--------|----------|
| `gwexpy.timeseries.__init__.__getattr__` | gwexpy/__init__.py | 37 | 削除/改名? |
| `gwexpy.timeseries.__init__.__dir__` | gwexpy/__init__.py | 41 | 削除/改名? |

## 2. 実コードにあるが台帳にない（台帳の漏れ）

| qualified_name | file | lineno | 推定原因 |
|----------------|------|--------|----------|
| `gwexpy.timeseries._signal.TimeSeriesSignalMixin.demodulate` | gwexpy/timeseries/_signal.py | 676 | 新規追加? |
| `gwexpy.timeseries.__getattr__` | gwexpy/timeseries/__init__.py | 37 | 新規追加? |
| `gwexpy.timeseries.__dir__` | gwexpy/timeseries/__init__.py | 41 | 新規追加? |

## 3. ファイルパス不一致

*なし*

## 4. 行番号の大幅乖離（±30行超）

| qualified_name | 台帳lineno | 実コードlineno | 差分 |
|----------------|------------|----------------|------|
| `gwexpy.timeseries._signal.TimeSeriesSignalMixin.baseband` | 674 | 760 | 86 |
| `gwexpy.timeseries._signal.TimeSeriesSignalMixin.lock_in` | 896 | 982 | 86 |
| `gwexpy.timeseries._signal.TimeSeriesSignalMixin.transfer_function` | 1126 | 1217 | 91 |
| `gwexpy.timeseries._signal.TimeSeriesSignalMixin.xcorr` | 1384 | 1475 | 91 |

---

## Summary

- 台帳にあるが実コードに存在しない: 2 件
- 実コードにあるが台帳にない: 3 件
- ファイルパス不一致: 0 件
- 行番号の大幅乖離: 4 件
