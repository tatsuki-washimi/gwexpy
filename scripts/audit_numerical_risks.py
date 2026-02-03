import ast
import os
import sys
from typing import Optional


def is_small_float(n: object) -> bool:
    if isinstance(n, float):
        return 0.0 < abs(n) <= 1e-4
    return False


RISKY_FUNCS = {
    # Linear Algebra
    "inv",
    "det",
    "solve",
    "pinv",
    "lstsq",
    # Signal Processing
    "welch",
    "csd",
    "coherence",
    "spectrogram",
    "lfilter",
    "filtfilt",
    # Fitting/Optimization
    "curve_fit",
    "leastsq",
    "minimize",
    # Polynomials
    "polyfit",
    "roots",
    # GW specific types (if they appear as calls)
    "TimeSeries",
    "FrequencySeries",
}

DANGEROUS_CATCHES = {
    "Exception",
    "BaseException",
    "ArithmeticError",
    "ZeroDivisionError",
    "FloatingPointError",
    "ValueError",
}


class DeepRiskVisitor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.risks = []

    def _log(self, node, category, detail):
        self.risks.append(
            {
                "file": self.filename,
                "line": node.lineno,
                "category": category,
                "detail": detail,
            }
        )

    # --- 1. Hardcoded Floats & Defaults ---
    def visit_FunctionDef(self, node):
        if node.args.defaults:
            args_with_defaults = node.args.args[-len(node.args.defaults) :]
            for arg, default in zip(args_with_defaults, node.args.defaults):
                self._check_default(node, arg.arg, default)

        if node.args.kw_defaults:
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if default is not None:
                    self._check_default(node, arg.arg, default)

        self.generic_visit(node)

    def _check_default(self, node, arg_name, value_node):
        val = self._extract_value(value_node)
        if val is not None and is_small_float(val):
            self._log(node, "Dangerous Default", f"Arg '{arg_name}' = {val}")

    def _extract_value(self, node: ast.AST) -> Optional[float]:
        # mypy: ast.Constant.value can be many types; we only care about numeric literals.
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            return -float(node.operand.value)
        return None

    def visit_Assign(self, node):
        val = self._extract_value(node.value)
        if val is not None and is_small_float(val):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            for t in targets:
                self._log(node, "Hardcoded Float", f"{t} = {val}")
        self.generic_visit(node)

    # --- 2. Risky Operations ---
    def visit_BinOp(self, node):
        # Check formatting "%.2f" % val
        if (
            isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
        ):
            import re

            match = re.search(r"%\.(\d+)f", node.left.value)
            if match:
                precision = int(match.group(1))
                if precision < 10:
                    self._log(
                        node, "Formatting Risk", f"Low precision format: %.{precision}f"
                    )

        # Addition/Sub of small floats
        if isinstance(node.op, (ast.Add, ast.Sub)):
            val = self._extract_value(node.left)
            if val is not None and is_small_float(val):
                self._log(node, "Risky Arithmetic", f"Op with {val}")
            val = self._extract_value(node.right)
            if val is not None and is_small_float(val):
                self._log(node, "Risky Arithmetic", f"Op with {val}")

        self.generic_visit(node)

    def visit_Compare(self, node):
        for comparator in node.comparators:
            val = self._extract_value(comparator)
            if val is not None and is_small_float(val):
                self._log(node, "Risky Comparison", f"Compare with {val}")
        self.generic_visit(node)

    # --- 3. Exception Swallowing ---
    def visit_Try(self, node):
        for handler in node.handlers:
            if handler.type:
                # Check what exception is caught
                exc_name = self._get_name(handler.type)
                if exc_name in DANGEROUS_CATCHES:
                    # Check if body is empty or just 'pass' or 'return None' or 'continue'
                    if self._is_swallowing(handler.body):
                        self._log(node, "Exception Swallowing", f"Silencing {exc_name}")
        self.generic_visit(node)

    def _get_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr  # Simplification
        elif isinstance(node, ast.Tuple):
            return "Tuple"  # Handling tuple catches later if needed
        return None

    def _is_swallowing(self, body):
        # If body is just 'pass', or 'return None', or return constant, log it.
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Continue):
                return True
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
                return True  # Returning literal (None or 0)
        return False

    # --- 4. Risky Function Calls ---
    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in RISKY_FUNCS:
            self._log(node, "Risky Function Call", f"Calling {func_name}")

        # Check for float32 casting
        if func_name in ("float32", "float16"):
            self._log(node, "Precision Risk", f"Cast to {func_name}")

        # Check np.array(..., dtype='float32')
        if (
            func_name == "array"
            or func_name == "zeros"
            or func_name == "ones"
            or func_name == "full"
        ):
            for kw in node.keywords:
                if kw.arg == "dtype":
                    if isinstance(kw.value, ast.Constant) and isinstance(
                        kw.value.value, str
                    ):
                        if "float32" in kw.value.value or "float16" in kw.value.value:
                            self._log(
                                node,
                                "Precision Risk",
                                f"Creation with dtype={kw.value.value}",
                            )
                    if isinstance(kw.value, ast.Attribute) and kw.value.attr in (
                        "float32",
                        "float16",
                    ):
                        self._log(
                            node,
                            "Precision Risk",
                            f"Creation with dtype={kw.value.attr}",
                        )

        self.generic_visit(node)

    # --- 5. Precision & Boolean Risks ---
    def visit_JoinedStr(self, node):
        # f-string checks: f"{val:.2f}"
        for value in node.values:
            if isinstance(value, ast.FormattedValue) and value.format_spec:
                # Check format spec for low precision
                self._check_format_spec(node, value.format_spec)
        self.generic_visit(node)

    def visit_Call_Formatting(self, node):
        # Check .format() calls - hard to parse args but we can check if string literal has pattern?
        # Simpler: check for Modulo formatting in BinOp
        pass

    def _check_format_spec(self, node, format_spec):
        # format_spec is a JoinedStr or Constant
        spec_str = ""
        if isinstance(format_spec, ast.Constant):
            if isinstance(format_spec.value, str):
                spec_str = format_spec.value
        elif isinstance(format_spec, ast.JoinedStr):
            # Simplified: just check constant parts
            for v in format_spec.values:
                if isinstance(v, ast.Constant):
                    spec_str += str(v.value)

        # Look for .Nf where N is small (e.g. .2f, .4f)
        # Regex or simple parsing.
        # " .2f " -> risk if data is 1e-21
        import re

        match = re.search(r"\.(\d+)f", spec_str)
        if match:
            precision = int(match.group(1))
            if (
                precision < 10
            ):  # 10 digits is reasonable safety, but typical 2 or 4 is bad for GW
                self._log(
                    node, "Formatting Risk", f"Low precision format: .{precision}f"
                )

    def visit_If(self, node):
        # Check implicit boolean: if x:
        self._check_boolean_condition(node, node.test)
        self.generic_visit(node)

    def visit_While(self, node):
        self._check_boolean_condition(node, node.test)
        self.generic_visit(node)

    def _check_boolean_condition(self, node, test_node):
        # Check if test_node is a Name that typically holds a float/array
        # This is heuristics based on names
        if isinstance(test_node, ast.Name):
            name = test_node.id
            if name in (
                "data",
                "series",
                "signal",
                "psd",
                "coherence",
                "value",
                "gain",
            ):
                self._log(node, "Implicit Boolean", f"Implicit check on '{name}'")

    def visit_BinOp_Formatting(self, node):
        # Handle "%.2f" % val
        if (
            isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
        ):
            import re

            match = re.search(r"%\.(\d+)f", node.left.value)
            if match:
                precision = int(match.group(1))
                if precision < 10:
                    self._log(
                        node, "Formatting Risk", f"Low precision format: %.{precision}f"
                    )


def scan_file(filepath):
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        visitor = DeepRiskVisitor(filepath)
        visitor.visit(tree)
        return visitor.risks
    except Exception:
        # print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return []


def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "gwexpy"
    all_risks = []

    for root, dirs, files in os.walk(target_dir):
        if "tests" in root.split(os.sep):
            continue
        if "gui/test-data" in root:
            continue

        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                risks = scan_file(path)
                all_risks.extend(risks)

    all_risks.sort(key=lambda x: (x["category"], x["file"], x["line"]))

    print(f"Scanned {target_dir}. Found {len(all_risks)} potential risks.\n")

    current_cat = ""
    for r in all_risks:
        if r["category"] != current_cat:
            current_cat = r["category"]
            print(f"\n--- {current_cat} ---")
        print(f"{r['file']}:{r['line']} : {r['detail']}")


if __name__ == "__main__":
    main()
