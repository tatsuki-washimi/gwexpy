#!/usr/bin/env bash

# audit_api.sh: リポジトリ内の全クラス・関数を抽出するスクリプト
# 使用方法: ./audit_api.sh [ターゲットディレクトリ]

TARGET_DIR=${1:-"gwexpy/"}

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR not found."
    exit 1
fi

echo "--- API Elements Audit for $TARGET_DIR ---"

# --- Classes ---
echo -e "\n[Classes]"
find "$TARGET_DIR" -name "*.py" ! -name "*test*" -exec grep -H "^[[:space:]]*class " {} + | \
    sed "s|^$TARGET_DIR/||" | sed -E "s/\.py:[[:space:]]*class[[:space:]]+([A-Za-z0-9_]+).*/: \1/" | sort | uniq

# --- Functions ---
echo -e "\n[Functions (Public Only)]"
find "$TARGET_DIR" -name "*.py" ! -name "*test*" -exec grep -H "^def " {} + | \
    grep -vE ":def[[:space:]]+_" | \
    sed "s|^$TARGET_DIR/||" | sed -E "s/\.py:def[[:space:]]+([A-Za-z0-9_]+).*/: \1/" | sort | uniq

# --- Summary Counts ---
echo -e "\n[Summary]"
CLASS_COUNT=$(find "$TARGET_DIR" -name "*.py" ! -name "*test*" -exec grep -c "^[[:space:]]*class " {} + | awk -F: '{sum += $2} END {print sum}')
FUNC_COUNT=$(find "$TARGET_DIR" -name "*.py" ! -name "*test*" -exec grep -c "^def " {} + | awk -F: '{sum += $2} END {print sum}')
PUBLIC_FUNC_COUNT=$(find "$TARGET_DIR" -name "*.py" ! -name "*test*" -exec grep -H "^def " {} + | grep -vE ":def[[:space:]]+_" | wc -l)

echo "Total Classes: $CLASS_COUNT"
echo "Total Top-level Functions: $FUNC_COUNT"
echo "Public Top-level Functions (no _ prefix): $PUBLIC_FUNC_COUNT"
