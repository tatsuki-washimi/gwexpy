from __future__ import annotations

import importlib
import unittest


class GravitySpyCompatibilityTests(unittest.TestCase):
    def test_io_wrapper_exports_current_gwpy_symbols(self) -> None:
        module = importlib.import_module("gwexpy.table.io.gravityspy")

        self.assertEqual(
            module.__all__,
            ["EventTable", "get_gravityspy_triggers"],
        )
        self.assertTrue(hasattr(module, "EventTable"))
        self.assertTrue(hasattr(module, "get_gravityspy_triggers"))

    def test_table_wrapper_still_exports_gravityspy_table(self) -> None:
        module = importlib.import_module("gwexpy.table.gravityspy")

        self.assertIn("GravitySpyTable", module.__all__)
        self.assertTrue(hasattr(module, "GravitySpyTable"))


if __name__ == "__main__":
    unittest.main()
