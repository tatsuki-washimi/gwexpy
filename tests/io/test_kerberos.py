import sys
import pytest

from gwpy.io.tests import test_kerberos as _gwpy_test_kerberos
from gwpy.io.tests.test_kerberos import *  # noqa: F403


def test_kinit_notty():
    if sys.stdout.isatty():
        pytest.skip("requires non-interactive stdout")
    try:
        _gwpy_test_kerberos.test_kinit_notty()
    except OSError:
        pytest.skip("stdin capture prevents interactive kerberos prompts")
