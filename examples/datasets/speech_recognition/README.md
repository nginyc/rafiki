# Material Preparation

Before you could run training using the TfDeepSpeech model, you will need to supply following materials to the model:

1. Training and testing datasets

2. N-gram Language Model

3. Trie (A keyword tree)


## Download and Preprocess the Dataset

Speech Recognition models in Rafiki require a specific dataset format, see [Supported Dataset Types](https://nginyc.github.io/rafiki/docs/latest/src/user/datasets.html#) for more details. Two examples scripts has been provided in this directory.

Do the following in `<rafiki_root_directory>`:

For testing purposes, you could run `python examples/datasets/speech_recognition/load_tiny_format.py` to download a dataset with only one voice sample.

For real training, you might want to use a voice corpus of considerable size. `python examples/datasets/speech_recognition/load_librispeech_format.py` helps you to load a benchmark dataset (~55GB) sufficient for training a functional Speech Recognition model.


## Use Pre-built LM and Trie

For English language Speech Recognition tasks, you can use pre-built Language Model and Trie to train your model.

The files are available at https://github.com/mozilla/DeepSpeech/tree/master/data/lm. You will need Git Large File Storage to clone the repository properly. Follow the instructions [here](https://git-lfs.github.com/) to install `git-lfs`.

Then clone the repository and copy the `lm.binary` and `trie` files into `<rafiki_root_directory>/data`, which is the default directory to store the two files as specified by the TfDeepSpeech model.

You are now ready to train your own TfDeepSpeech model!

If you wish to generate your own language models and trie files instead, or wish to implement TfDeepSpeech to other languages, see instructions provided below.


## Generate Language Models

The TfDeepSpeech model requires a binary n-gram language model compiled by `kenlm` to make predictions. Follow the steps in the example below to generate a LibriSpeech language model for English language:

1. Download the required txt.gz by running the python script

    ```sh 
    python examples/datasets/speech_recognition/load_lm_txt.py
    ```

1. Install dependencies

    ```sh
    sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
    ```

2. Build KenLM Language Model Toolkit

    ```sh
    cd /tmp/
    wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
    mkdir kenlm/build
    cd kenlm/build
    cmake ..
    make -j2
    ```
3. Build pruned LM

    ```sh
    bin/lmplz --order 5 \
              --temp_prefix /tmp/ \
              --memory 50% \
              --text /tmp/lower.txt \
              --arpa /tmp/lm.arpa \
              --prune 0 0 0 1
    ```
    This step may take an hour to complete.
    
4. Quantize and produce trie binary

    Now substitute `<rafiki_root_directory>` with the path to rafiki root, and run the following:

    ```sh
    bin/build_binary -a 255 \
                     -q 8 \
                     trie \
                     /tmp/lm.arpa \
                     <rafiki_root_directory>/data/lm.binary
    rm /tmp/lm.arpa
    ```
    The `lm.binary` binary Language Model file is now in the data directory.
    
## Generate Trie 

See documentation on [DeepSpeech Git Repo](https://github.com/mozilla/DeepSpeech/tree/master/native_client) to generate the trie for your language model. Follow the steps up to **Compile libdeepspeech.so & generate_trie** section. The generated binaries will be saved to `bazel-bin/native-client/`.

Remember to modify the `alphabet.txt` file if you are training TfDeepSpeech on languages other than English.

Run

```sh
bazel-bin/native-clinet/generate_trie ../rafiki/examples/datasets/speech_recognition/alphabet.txt ../rafiki/data/lm.binary ../rafiki/data/trie
```

The `trie` file is now in the data directory.

*Note: The `generate_trie` binaries are subject to updates by the DeepSpeech team. If you find mismatch of trie file version, update the version of ctc_decoder package by amending the `VERSION` variable in `examples/models/speech_recognition/utils/taskcluster.py`.*