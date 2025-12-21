Quickstart
==========

Create and analyze a simple time series.

.. code-block:: python

   import numpy as np
   from gwexpy.timeseries import TimeSeries

   ts = TimeSeries(np.random.randn(4096), t0=0, dt=1/1024, name="demo")
   bandpassed = ts.bandpass(30, 300)
   spectrum = bandpassed.fft()

   print(spectrum.frequencies[:5])

Working with collections:

.. code-block:: python

   from gwexpy.timeseries import TimeSeriesDict

   tsd = TimeSeriesDict()
   tsd["H1:TEST"] = ts
   tsd["L1:TEST"] = ts * 0.5

   matrix = tsd.to_matrix()
   print(matrix.shape)
