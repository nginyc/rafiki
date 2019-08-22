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

.. _`dataset-type:AUDIO_FILES`:

AUDIO_FILES
--------------------------------------------------------------------

The dataset file must be of the ``.zip`` archive format with a ``audios.csv`` at the root of the directory.
