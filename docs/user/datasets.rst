Supported Dataset Types
====================================================================

.. contents:: Table of Contents

Dataset URIs must have the protocols of either ``http`` or ``https``.

.. note::
    
    You can alternatively use relative or absolute filepaths as dataset URIs, only if you have deployed the full Rafiki stack on your own machine.

.. note::

    Refer to `./examples/datasets/ <https://github.com/nginyc/rafiki/tree/master/examples/datasets/>`_ for examples on pre-processing 
    common dataset formats to conform to the Rafiki's own dataset formats.


.. _`dataset-type:IMAGE_FILES`:

IMAGE_FILES
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format with a ``images.csv`` at the root of the directory.

The ``images.csv`` should be of a .CSV format with 2 columns of ``path`` and ``class``.

For each row,

    ``path`` should be a file path to a ``.png``, ``.jpg`` or ``.jpeg`` image file within the archive, relative to the root of the directory.

    ``class`` should be an integer from ``0`` to ``k - 1``, where ``k`` is the number of classes in the classification of images.

An example of ``images.csv`` follows:

.. code-block:: text

    path,class
    image-0-of-class-0.png,0
    image-1-of-class-0.png,0
    ...
    image-0-of-class-1.png,1
    ...
    image-99-of-class-9.png,9
    
