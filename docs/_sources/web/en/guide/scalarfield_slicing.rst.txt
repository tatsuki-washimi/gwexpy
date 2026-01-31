ScalarField Slicing Guide
=========================

4D Structure Preservation in ScalarField
-----------------------------------------

``ScalarField`` always maintains a 4D structure, even after indexing operations. This differs from the standard behavior of NumPy arrays and GWpy.

4D Structure Preservation
~~~~~~~~~~~~~~~~~~~~~~~~~~

**NumPy array behavior**:

.. code-block:: python

   >>> import numpy as np
   >>> arr = np.zeros((10, 5, 5, 5))
   >>> arr[0].shape
   (5, 5, 5)  # Dimension reduced

**ScalarField behavior**:

.. code-block:: python

   >>> from gwexpy.fields import ScalarField
   >>> field = ScalarField(np.zeros((10, 5, 5, 5)), ...)
   >>> field[0].shape
   (1, 5, 5, 5)  # 4D structure preserved

This behavior provides the following benefits:

1. **Axis metadata retention**: ``axis0_domain``, ``space_domain``, etc. are preserved
2. **Broadcast operation consistency**: Always treated as 4D
3. **FFT operation safety**: Can perform FFT while maintaining domain information

Reducing Dimensions
~~~~~~~~~~~~~~~~~~~~

To explicitly reduce dimensions, use the ``squeeze()`` method:

.. code-block:: python

   >>> field[0].squeeze().shape
   (5, 5, 5)  # Length-1 axes removed

Slicing Examples
~~~~~~~~~~~~~~~~

.. code-block:: python

   >>> # Extract specific time snapshot
   >>> snapshot = field[100]  # shape: (1, 5, 5, 5)

   >>> # Extract spatial cross-section
   >>> plane = field[:, :, :, 2]  # shape: (n_time, 5, 5, 1)

   >>> # Extract time series at specific spatial point
   >>> point_ts = field[:, 2, 2, 2]  # shape: (n_time, 1, 1, 1)
   >>> # Use squeeze for TimeSeries-like handling
   >>> point_ts_1d = point_ts.squeeze()  # shape: (n_time,)

Frequently Asked Questions (FAQ)
---------------------------------

Why preserve 4D structure?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

ScalarField represents physical quantities with different domains (time/frequency, real space/k-space) per axis. Reducing dimensions would lose this metadata, causing FFT operations and domain transformations to fail.

What if I need NumPy-like behavior?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``squeeze()`` method. This removes length-1 axes and returns a NumPy-like array.

Troubleshooting
---------------

Broadcast operations not working as expected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ScalarField is always 4D, so you need to match shapes when operating with other arrays.

.. code-block:: python

   # Incorrect: Operation with 1D array
   field + np.array([1, 2, 3])  # Error

   # Correct: Match shape
   field + np.array([1, 2, 3]).reshape(3, 1, 1, 1)

Related Links
-------------

- :doc:`../tutorials/field_scalar_intro` - ScalarField Tutorial
- :doc:`../reference/ScalarField` - ScalarField API Reference
- :doc:`../reference/FieldList` - FieldList / FieldDict Collections
