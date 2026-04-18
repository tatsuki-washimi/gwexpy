.. meta::
   :description: GWexpy reference hub for stable API, class, and topic lookups, with entry points for modules, class names, and theory notes.

.. _reference-en-entry:

Reference
=========

.. note::
   Page role: Reference index

**Stability:** Stable

Use this page as the hub for navigating the GWexpy reference. Choose the entry point that matches how you are looking things up: by module, by class name, or by concept/topic.

**Audience:** Users who already know the feature they need and want exact API, class, or topic details.
**Prerequisites:** Basic familiarity with GWexpy terminology and at least one guide or tutorial path.
**Use this page for:** Jumping into stable lookup-oriented documentation rather than task-oriented learning.
**Search hints:** API index, class index, topics, reference, module lookup, class lookup, theory notes

.. note::
   On this page:
   Use API Index for subsystem browsing, Class Index for Python class names, and Topics for conventions, theory, and helper material.

.. note::
   Advanced/theory landing:
   For validation assumptions, conventions, and audit-backed theory notes, start with :doc:`topics` and then continue to :doc:`../user_guide/validated_algorithms`.

.. note::
   Example framing:
   Goal: find the exact object, module, or topic page behind a name you already know.
   Inputs: a class name, module family, or concept.
   Outputs: a stable reference entry point for deeper lookup.

.. _reference-en-entry-table:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Entry Point
     - Stability
     - Use it for
   * - :doc:`API Index <api/index>`
     - Stable
     - Browsing modules and public functions by subsystem
   * - :doc:`Class Index <classes>`
     - Stable
     - Finding class pages by Python class name
   * - :doc:`Topics <topics>`
     - Stable
     - Finding theory, conventions, and helper pages by concept

.. _reference-en-entry-cards:

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 🧩 API Index
        :link: api/index
        :link-type: doc

        Browse module and function reference pages by subsystem.

    .. grid-item-card:: 🏗️ Class Index
        :link: classes
        :link-type: doc

        Find major GWexpy classes in alphabetical order.

    .. grid-item-card:: 🧭 Topics
        :link: topics
        :link-type: doc

        Start from conventions, theory notes, and helper overviews.

.. seealso::
   Hub links:

   - :doc:`api/index` for subsystem-by-subsystem API browsing
   - :doc:`topics` for the theory/conventions landing and bridge pages
   - :doc:`../user_guide/tutorials/index` for tutorial-first navigation before dropping into reference details

.. note::
   If you want task-oriented learning material rather than the reference, jump directly to the matching guide:

   - :doc:`../user_guide/scalarfield_slicing` for `ScalarField` slicing and field-style workflows
   - :doc:`../user_guide/validated_algorithms` for advanced validation assumptions and audit-backed theory notes
   - :doc:`../user_guide/gwexpy_for_gwpy_users_en` for GWpy migration-oriented usage

.. seealso::
   Next to read:

   - :doc:`../user_guide/tutorials/index` for notebook-driven learning by feature
   - :doc:`api/index` for detailed module/category entry points
   - :doc:`topics` if you want theory, conventions, and helper pages grouped by concept

.. toctree::
   :maxdepth: 2

   api/index
   classes
   topics
