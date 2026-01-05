
import sys
import os
from pathlib import Path

# Add repo root to path
sys.path.append(os.getcwd())

try:
    from gwexpy.io.dttxml_common import load_dttxml_products
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_load_all_xmls():
    xml_files = [
        "gwexpy/gui/SPEC_INJ_82Hz_OMCPZT.xml",
        "gwexpy/gui/TS_ETMX_23-6kHz_ringdown_No2_20251202.xml",
        "gwexpy/gui/test.xml"
    ]

    for f in xml_files:
        path = Path(f)
        if not path.exists():
            print(f"Skipping {f} (Not found)")
            continue

        print(f"\n--- Testing {f} ---")
        try:
            products = load_dttxml_products(str(path))
            if not products:
                print("  [WARN] No products loaded (empty dict returned).")
            else:
                for prod_type, content in products.items():
                    print(f"  Type: {prod_type}")
                    # Content is a dict of channel/pair -> info dict
                    count = 0
                    for key, val in content.items():
                        count += 1
                        data = val.get('data', [])
                        shape = data.shape if hasattr(data, 'shape') else len(data)
                        print(f"    Item: {key}, Data Shape: {shape}")
                        if count >= 3:
                            print("    ... (more items)")
                            break
        except Exception as e:
            print(f"  [FAIL] Error loading: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_load_all_xmls()
