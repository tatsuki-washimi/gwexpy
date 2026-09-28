"""Deterministic F3 PSD metadata fault extension to the frozen public oracle."""
import copy
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

FROZEN = Path('/tmp/gwexpy-v025-f3-frozen-harness/benchmarks/io')
sys.path.insert(0, str(FROZEN))
from f3_dttxml_fixtures import write_fixtures


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    base = write_fixtures(root / 'base', unselected_results=1, points=8)
    original = Path(base['cases']['many_valid']['path'])
    variants = {
        't0_nan': ('t0', 'nan'), 't0_inf': ('t0', 'inf'),
        'f0_nan': ('f0', 'nan'), 'f0_inf': ('f0', 'inf'),
        'f0_negative': ('f0', '-1'), 'df_nan': ('df', 'nan'),
        'df_inf': ('df', 'inf'), 'df_negative': ('df', '-1'),
        'df_zero': ('df', '0'), 'bunit_invalid': ('BUnit', 'bogus_unit'),
        'encoding_invalid': ('Encoding', 'NoEndian,base64'),
        'large_n': ('N', '65537'),
    }
    cases = {}
    for name, (field, value) in variants.items():
        tree = ET.parse(original)
        result = tree.getroot().findall('LIGO_LW')[1]
        if field == 't0':
            result.find("Time[@Name='t0']").text = value
        elif field == 'Encoding':
            result.find('Array/Stream').set('Encoding', value)
        else:
            param = result.find(f"Param[@Name='{field}']")
            if param is None:
                param = ET.SubElement(result, 'Param', {'Name': field})
            param.text = value
        if field == 'N':
            result.find('Array').findall('Dim')[1].text = value
        path = root / f'{name}.xml'
        tree.write(path, encoding='utf-8', xml_declaration=True)
        case = copy.deepcopy(base['cases']['many_valid'])
        case.update(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                    oracle_category=f'unselected_metadata_{name}', native_parser_exception_applies=False)
        cases[name] = case
    tree = ET.parse(original)
    result = tree.getroot().findall('LIGO_LW')[1]
    result.find("Param[@Name='f0']").text = '1e308'
    result.find("Param[@Name='df']").text = '1e308'
    path = root / 'frequency_overflow.xml'
    tree.write(path, encoding='utf-8', xml_declaration=True)
    case = copy.deepcopy(base['cases']['many_valid'])
    case.update(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                oracle_category='unselected_metadata_frequency_overflow', native_parser_exception_applies=False)
    cases['frequency_overflow'] = case
    for mode in ('warn', 'raise'):
        case = copy.deepcopy(base['cases']['many_valid'])
        case.update(oracle_category=f'underflow_mode_{mode}', underflow_mode=mode,
                    native_parser_exception_applies=False)
        cases[f'underflow_{mode}'] = case
    manifest = {'schema':'gwexpy-f3-psd-metadata-extension-v1',
                'frozen_fixture_helper_sha256':sha(FROZEN / 'f3_dttxml_fixtures.py'),
                'generator_sha256':sha(Path(__file__)), 'cases':cases}
    (root / 'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print(root / 'manifest.json')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
