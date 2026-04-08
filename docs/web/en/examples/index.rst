Case Studies
============

A collection of practical workflows targeting real-world analysis tasks.
To learn basic operations for each feature (arguments, return values, etc.), please refer to :doc:`../user_guide/tutorials/index`.

Case Study Gallery
------------------

.. grid:: 1
    :gutter: 3

    .. grid-item-card:: 📈 Noise Budget Analysis
        :link: ../user_guide/tutorials/case_noise_budget
        :link-type: doc

        *   **Problem**: Identify major noise sources in observation data.
        *   **Approach**: Multi-channel coherence analysis and spectral synthesis.
        *   **Key APIs**: ``TimeSeriesMatrix``, ``PSD``, ``Coherence``.

    .. grid-item-card:: 🎛️ Transfer Function Measurement & Fitting
        :link: ../user_guide/tutorials/case_transfer_function
        :link-type: doc

        *   **Problem**: Measure system transfer functions and compare with theoretical models.
        *   **Approach**: TF measurement and pole-zero placement using sine-sweeps or white-noise excitation.
        *   **Key APIs**: ``TransferFunction``, ``Fitter``, ``BodePlot``.

    .. grid-item-card:: 🏗️ Active Damping Control
        :link: ../user_guide/tutorials/case_active_damping
        :link-type: doc

        *   **Problem**: Design and evaluate MIMO control systems to suppress suspension resonances.
        *   **Approach**: Feedback control simulation using state-space models.
        *   **Key APIs**: ``StateSpaceMatrix``, ``ActiveControl``, ``LQR``.

    .. grid-item-card:: ✂️ Segment Analysis of Long-term Data
        :link: ../user_guide/tutorials/case_segment_analysis
        :link-type: doc

        *   **Problem**: Extract specific intervals (segments) satisfying conditions from multi-day data for statistical processing.
        *   **Approach**: Data querying and parallel processing using ``SegmentTable``.
        *   **Key APIs**: ``SegmentTable``, ``SegmentList``, ``Fetch``.

.. toctree::
   :hidden:

   ../user_guide/tutorials/case_noise_budget
   ../user_guide/tutorials/case_transfer_function
   ../user_guide/tutorials/case_active_damping
   ../user_guide/tutorials/case_segment_analysis
