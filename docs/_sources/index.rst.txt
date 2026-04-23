GWexpy
======

.. raw:: html

   <style>
   /* Gateway Specific: Hide sidebar and make full wide */
   .wy-nav-side          { display: none !important; }
   .wy-nav-content-wrap  { margin-left: 0 !important; }
   .wy-nav-content       { max-width: 900px !important; margin: 0 auto !important; }
   </style>

.. raw:: html

   <div class="gw-hero" style="
     text-align: center;
     padding: 3em 1em 2em;
   ">
     <p style="font-size:1.25em; color:#555; margin-bottom:0.5em;">
       Extended Python Toolkit for Gravitational-Wave Data Analysis
     </p>
     <p style="font-size:1.05em; color:#777; margin-top:0;">
       A multi-dimensional data analysis library extending GWpy — 
       integrating matrices, fields, fitting, and advanced signal processing.
     </p>
   </div>

.. grid:: 2
   :gutter: 3
   :class-container: gw-cta-grid

   .. grid-item::

      .. button-ref:: web/en/index
         :ref-type: doc
         :color: primary
         :shadow:
         :expand:

         📖 English Documentation

   .. grid-item::

      .. button-ref:: web/ja/index
         :ref-type: doc
         :color: secondary
         :shadow:
         :expand:

         📖 日本語ドキュメント

----

Key Features
------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: 🔬 Multidimensional Fields
      :text-align: center

      ``ScalarField`` / ``VectorField`` / ``TensorField``

      Uniform interface for multidimensional data across space and time.

   .. grid-item-card:: ⚡ Numerical Stability
      :text-align: center

      Safe Log / Zero-division protection / NaN propagation detection

      Automatic robustness for mission-critical scientific computing.

   .. grid-item-card:: 📊 Integrated Analysis Tools
      :text-align: center

      BrUCo / ARIMA / Fitting / MCMC

      Seamless transition from noise characterization to advanced fitting.

----

Quick Installation
------------------

.. code-block:: bash

   git clone https://github.com/tatsuki-washimi/gwexpy.git
   cd gwexpy && pip install -e .

Quick Demo
----------

.. code-block:: python

   from gwexpy.timeseries import FrequencySeriesMatrix
   fsmtx = FrequencySeriesMatrix.read("data.hdf5")
   fsmtx.fit(model="lorentzian").plot()

.. image:: _static/images/hero_plot.png
   :alt: FrequencySeriesMatrix Fitting Output
   :align: center
   :width: 100%

.. toctree::
   :hidden:

   web/en/index
   web/ja/index
