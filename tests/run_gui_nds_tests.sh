#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "ERROR: xvfb-run not found. Install Xvfb/xvfb-run or run without headless mode." >&2
  exit 127
fi

# Set sensible defaults for KAGRA environment if unset
export GWEXPY_NDS_CHANNELS="${GWEXPY_NDS_CHANNELS:-K1:PEM-MIC_OMC_BOOTH_OMC_Z_OUT_DQ}"
export GWEXPY_ENABLE_NDS_TESTS="${GWEXPY_ENABLE_NDS_TESTS:-1}"

echo "--- NDS Test Environment ---"
echo "GWEXPY_ENABLE_NDS_TESTS: ${GWEXPY_ENABLE_NDS_TESTS}"
echo "GWEXPY_NDS_HOST: ${GWEXPY_NDS_HOST:-[unset]}"
echo "GWEXPY_NDS_PORT: ${GWEXPY_NDS_PORT:-[unset]}"
echo "GWEXPY_NDS_CHANNELS: ${GWEXPY_NDS_CHANNELS}"
echo "GWEXPY_NDS_DURATION: ${GWEXPY_NDS_DURATION:-[unset]}"
echo "NDSSERVER: ${NDSSERVER:-[unset]}"
echo "NDS2SERVER: ${NDS2SERVER:-[unset]}"
echo "----------------------------"

if [ "${GWEXPY_ENABLE_NDS_TESTS}" != "1" ]; then
  echo "NDS tests are disabled (GWEXPY_ENABLE_NDS_TESTS != 1). Skipping."
  exit 0
fi

echo "Checking NDS test prerequisites..."
PYTHONPATH=".:${PYTHONPATH:-}" python3 -c "
import os
import sys
try:
    import nds2
    print('NDS2 client: OK')
except ImportError:
    print('NDS2 client: MISSING', file=sys.stderr)
    sys.exit(0)

# Preferred endpoint resolution logic
host = os.getenv('GWEXPY_NDS_HOST')
port = os.getenv('GWEXPY_NDS_PORT')
nds1_server = os.getenv('NDSSERVER')
nds2_server = os.getenv('NDS2SERVER')

endpoints = []
if host and port:
    endpoints.append(f'{host}:{port}')

if nds1_server:
    endpoints.extend([e.strip() for e in nds1_server.split(',')])

if nds2_server:
    endpoints.extend([e.strip() for e in nds2_server.split(',')])

if not endpoints:
    endpoints = ['nds.ligo.caltech.edu:31200']

success = False
for endpoint in endpoints:
    try:
        if ':' in endpoint:
            h, p = endpoint.split(':', 1)
            p = int(p)
        else:
            h, p = endpoint, 31200 # Default if no port
        
        import socket
        socket.setdefaulttimeout(5)
        conn = nds2.connection(h, p)
        # Verify if we can find the channel (optional but good)
        # For KAGRA channels, we often need to check both online and static
        print(f'Trying NDS Server {h}:{p}...')
        conn.close()
        print(f'NDS Server ({h}:{p}): OK')
        success = True
        break
    except Exception as e:
        print(f'NDS Server ({endpoint}): UNREACHABLE or ERROR ({e})', file=sys.stderr)

if not success:
    sys.exit(1)
" || { echo "Prerequisites check failed. Tests will likely be skipped."; }

echo "Running GUI NDS Integration Tests..."
xvfb-run -a -s "-screen 0 1920x1080x24" env PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 GWEXPY_ENABLE_NDS_TESTS=1 \
  python -u -m pytest -p pytestqt.plugin -m "nds and gui" -v
