
.. _`installing-python`:

Installing Python
====================================================================

Usage of Rafiki requires Python 3.6. Specifically, you'll need the command ``python`` to point to a Python 3.6 program, and ``pip`` to point to PIP for that Python 3.6 installation.

To achieve this, we recommend using *Conda* with a Python 3.6 environment as per the instructions below:

    1. Install the latest version of `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_

    2. Run the following commands on shell:

        .. code-block:: shell

            conda create --name rafiki python=3.6

    3. Every time you need to use ``python`` or ``pip`` for Rafiki, run the following command on shell:

        .. code-block:: shell

            conda activate rafiki


Otherwise, you can refer to these links below on installing Python natively: 

    - `Installing Python 3.6 for Ubuntu <http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/>`_
    - `Python download page for MacOS <https://www.python.org/downloads/mac-osx/>`_
    - `Using aliases to set the correct 'python' program <https://askubuntu.com/questions/320996/how-to-make-python-program-command-execute-python-3>`_
    - `Installing the correct PIP for 'python' <https://stackoverflow.com/questions/38938205/how-to-override-the-pip-command-to-python3-x-instead-of-python2-7>`_
