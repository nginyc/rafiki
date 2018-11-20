if ! [ -z "$1" ]
then
    RAFIKI_VERSION=$1
fi

pip install sphinx sphinx_rtd_theme
sphinx-build -b html . docs/$RAFIKI_VERSION/