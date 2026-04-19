Preprocessing
=============

**Stability:** Stable

.. seealso::

   :ref:`validated-en-adaptive-whitening`
      Validation assumptions and evidence for adaptive whitening and automatic stabilization.
   :doc:`../../user_guide/numerical_stability`
      Stability guidance for ``eps="auto"`` and related preprocessing defaults.
   :doc:`../../user_guide/tutorials/case_ml_preprocessing`
      Example preprocessing workflow that uses whitening in context.
   :doc:`../../user_guide/tutorials/ml_preprocessing_methods`
      Method-by-method guide for preprocessing utilities.
   :doc:`../../user_guide/tutorials/case_wiener_filter`
      Related case study that applies preprocessing choices inside a noise-subtraction workflow.

Preprocessing utilities are provided under ``gwexpy.signal.preprocessing``.

Main components:

- ``MLPreprocessor``
- ``standardize`` / ``StandardizationModel``
- ``whiten`` / ``WhiteningModel``
- ``impute``

See tutorial usage examples in:

- :doc:`../../user_guide/tutorials/case_ml_preprocessing`
- :doc:`../../user_guide/tutorials/ml_preprocessing_methods`
