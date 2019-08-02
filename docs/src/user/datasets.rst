Dataset Types
====================================================================

.. note::

    Refer to `./examples/datasets/ <https://github.com/nginyc/rafiki/tree/master/examples/datasets/>`_ for examples on pre-processing 
    common dataset formats to conform to the Rafiki's own dataset formats.

.. _`dataset-type:IMAGE_FILES`:

IMAGE_FILES
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format with a ``images.csv`` at the root of the directory.

The ``images.csv`` should be of a `.CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`_
format with columns of ``path`` and ``N`` other variable column names (*tag columns*).

For each row,

    ``path`` should be a file path to a ``.png``, ``.jpg`` or ``.jpeg`` image file within the archive, 
    relative to the root of the directory.

    The other ``N`` columns describe the corresponding image, *depending on the task*.


.. _`dataset-type:CORPUS`:

CORPUS
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format with a ``corpus.tsv`` at the root of the directory.

The ``corpus.tsv`` should be of a `.TSV <https://en.wikipedia.org/wiki/Tab-separated_values>`_ 
format with columns of ``token`` and ``N`` other variable column names (*tag columns*).

For each row,

    ``token`` should be a string, a token (e.g. word) in the corpus. 
    These tokens should appear in the order as it is in the text of the corpus.
    To delimit sentences, ``token`` can be take the value of ``\n``.

    The other ``N`` columns describe the corresponding token as part of the text of the corpus, *depending on the task*.

.. _`dataset-type:TABULAR`:

TABULAR
--------------------------------------------------------------------

The dataset file must be a tabular dataset of the ``.csv`` format with ``N`` columns.

An example of the dataset for the task ``TABULAR_REGRESSION`` follows:

.. code-block:: text

    density,bodyfat,age,weight,height,neck,chest,abdomen,hip,thigh,knee,ankle,biceps,forearm,wrist
    1.0708,12.3,23,154.25,67.75,36.2,93.1,85.2,94.5,59,37.3,21.9,32,27.4,17.1
    1.0853,6.1,22,173.25,72.25,38.5,93.6,83,98.7,58.7,37.3,23.4,30.5,28.9,18.2
    1.0414,25.3,22,154,66.25,34,95.8,87.9,99.2,59.6,38.9,24,28.8,25.2,16.6
    ...