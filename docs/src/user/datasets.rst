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

 The ``images.csv`` should be of a `.CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`_
format with 3 columns of ``wav_filename``, ``wav_filesize`` and ``transcript``.

 For each row,

     ``wav_filename`` should be a file path to a ``.wav`` audio file within the archive, relative to the root of the directory.

     ``wav_filesize`` should be an integer representing the size of the ``.wav`` audio file, in number of bytes.

     ``transcript`` should be a string of the true transcript for the audio file.

 An example of ``audios.csv`` follows:

 .. code-block:: text

    wav_filename,wav_filesize,transcript
    6930-81414-0000.wav,412684,audio transcript one
    6930-81414-0001.wav,559564,audio transcript two
    ...
    672-122797-0005.wav,104364,audio transcript one thousand
    ...
    1995-1837-0001.wav,279404,audio transcript three thousand
    