# Speech Recognition Models

## Customizing `TfDeepSpeech`

### Using Pre-built LM and Trie

By default, the `TfDeepSpeech` model should use the pre-built language model (LM) and trie from https://github.com/mozilla/DeepSpeech/tree/v0.6.0-alpha.4/data/lm for training,
which works for the *English* language. You'll need to first download this model's file dependencies by running (in Rafiki's root folder):
    
```
bash examples/models/speech_recognition/tfdeepspeech/download_file_deps.sh
```

This downloads the files `alphabet.txt`, `lm.binary` and `trie` into `<rafiki_root_directory>/tfdeepspeech`, where the `TfDeepSpeech` model reads its dependencies from by default.

If you wish to generate your own language models and trie files instead, or wish to implement TfDeepSpeech to other languages, see instructions provided below.

### Generating Language Models

The TfDeepSpeech model requires a binary n-gram language model compiled by `kenlm` to make predictions. Follow the steps in the example below to generate a LibriSpeech language model for English language:

1. Download the required txt.gz by running the python script

    ```sh 
    python examples/models/speech_recognition/tfdeepspeech/download_lm_txt.py
    ```

1. Install dependencies

    ```sh
    sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
    ```

2. Build KenLM Language Model Toolkit

    ```sh
    cd /tmp/
    wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
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
    
### Generating Trie 

See documentation on [DeepSpeech Git Repo](https://github.com/mozilla/DeepSpeech/tree/master/native_client) to generate the trie for your language model. Follow the steps up to **Compile libdeepspeech.so & generate_trie** section. The generated binaries will be saved to `bazel-bin/native-client/`.

Remember to modify the `alphabet.txt` file if you are training TfDeepSpeech on languages other than English.

Run

```sh
bazel-bin/native-clinet/generate_trie ../rafiki/examples/datasets/speech_recognition/alphabet.txt ../rafiki/data/lm.binary ../rafiki/data/trie
```

The `trie` file is now in the data directory.

*Note: The `generate_trie` binaries are subject to updates by the DeepSpeech team. If you find mismatch of trie file version, update the version of ctc_decoder package by amending the `VERSION` variable in `examples/models/speech_recognition/utils/taskcluster.py`.*