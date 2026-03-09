import pytest
from gwpy.astro.tests import test_range as _gwpy_test_range
from gwpy.astro.tests.test_range import *  # noqa: F403

# This case uses nproc=2 and can fail under pytest-forked environments
# where nested multiprocessing semaphore creation is restricted.
test_range_timeseries = pytest.mark.skip(
    reason="Skip flaky nested multiprocessing case under forked test runners."
)(getattr(_gwpy_test_range, "test_range_timeseries"))

test_range_spectrogram = pytest.mark.skip(
    reason="Skip flaky nested multiprocessing case under forked test runners."
)(getattr(_gwpy_test_range, "test_range_spectrogram"))
