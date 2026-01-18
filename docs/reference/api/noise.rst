Noise
=====

.. automodule:: gwexpy.noise
   :members:
   :undoc-members:
   :show-inheritance:

ASD
---

.. automodule:: gwexpy.noise.asd
   :members:
   :undoc-members:
   :show-inheritance:

Waveform
--------

Notes
~~~~~

``from_asd`` returns a ``TimeSeries`` (not a NumPy array). The output
inherits ``name`` and ``channel`` from the input ASD and uses
``unit * sqrt(Hz)`` as the time-domain unit. Use the ``t0`` argument
to set the start time.

.. automodule:: gwexpy.noise.wave
   :members:
   :undoc-members:
   :show-inheritance:
