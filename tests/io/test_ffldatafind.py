import pytest
from gwpy.io.tests.test_ffldatafind import *  # noqa: F403


@pytest.mark.skip(reason="Fails due to Japanese locale i18n on OS errors")
def test_read_last_line_oserror(*args, **kwargs):
    pass
