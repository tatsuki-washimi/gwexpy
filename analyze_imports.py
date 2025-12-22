import os
import ast
import sys
import pkgutil

def get_imports(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), path)
        except SyntaxError:
            return set()
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def is_stdlib(module_name):
    if module_name in sys.builtin_module_names:
        return True
    # For Python 3.10+
    if hasattr(sys, 'stdlib_module_names'):
        return module_name in sys.stdlib_module_names
    
    # Fallback for generic check (not perfect)
    import distutils.sysconfig as sysconfig
    std_lib = sysconfig.get_python_lib(standard_lib=True)
    try:
        spec = __import__(module_name).__spec__
        if spec and spec.origin:
             return std_lib in spec.origin
    except ImportError:
        pass
    return False

root_dir = 'gwexpy'
all_imports = set()
import_locations = {}

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.py'):
            filepath = os.path.join(dirpath, filename)
            imps = get_imports(filepath)
            for imp in imps:
                if imp == 'gwexpy': continue
                all_imports.add(imp)
                if imp not in import_locations:
                    import_locations[imp] = []
                import_locations[imp].append(filepath)

# Filter stdlib
stdlib_modules = {
    'sys', 'os', 're', 'math', 'time', 'datetime', 'json', 'collections', 'itertools', 'functools', 
    'warnings', 'logging', 'typing', 'abc', 'enum', 'pathlib', 'shutil', 'glob', 'tempfile', 'unittest',
    'subprocess', 'io', 'pickle', 'struct', 'copy', 'contextlib', 'inspect', 'traceback', 'gzip', 'tarfile',
    'zipfile', 'argparse', 'urllib', 'http', 'ftplib', 'smtplib', 'email', 'xml', 'html', 'concurrent',
    'multiprocessing', 'threading', 'queue', 'select', 'mmap', 'signal', 'socket', 'ssl', 'unicodedata',
    'operator', 'csv', 'sqlite3', 'ctypes', 'weakref', 'gc', 'platform', 'site', 'importlib', 'codecs',
    'numbers', 'decimal', 'fractions', 'random', 'statistics', 'colorsys', 'binascii', 'hashlib', 'hmac',
    'uuid', 'secrets', 'base64', 'calendar', 'textwrap', 'difflib', 'pprint', 'doctest', 'pdb', 'profile',
    'cProfile', 'timeit', 'trace', 'tracemalloc', 'distutils', 'ensurepip', 'venv', 'zipapp', '__future__'
}

external_imports = []
for imp in sorted(all_imports):
    if imp in stdlib_modules:
        continue
    # Simple check if installed/standard - heuristic
    external_imports.append(imp)

print("Detected external imports (candidates):")
for imp in external_imports:
    locations = import_locations[imp]
    print(f"{imp}: used in {len(locations)} files")
    # Show first few locations
    for loc in locations[:3]:
        print(f"  - {loc}")
