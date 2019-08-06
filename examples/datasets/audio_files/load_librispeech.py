#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# 

import os
import codecs
import fnmatch
import pandas
import tarfile
import shutil
import re
import unicodedata
from sox import Transformer
from tensorflow.python.platform import gfile
from examples.datasets.utils import download_dataset_from_url

LIBRIVOX_DIR = "LibriSpeech"
part_to_url = {
    'train-clean-100':  'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
    'train-clean-360':  'http://www.openslr.org/resources/12/train-clean-360.tar.gz',
    'train-other-500':  'http://www.openslr.org/resources/12/train-other-500.tar.gz',
    'dev-clean': 'http://www.openslr.org/resources/12/dev-clean.tar.gz',
    'dev-other': 'http://www.openslr.org/resources/12/dev-other.tar.gz',
    'test-clean': 'http://www.openslr.org/resources/12/test-clean.tar.gz',
    'test-other': 'http://www.openslr.org/resources/12/test-other.tar.gz',
}

def load_librispeech(data_dir, parts=['dev-clean', 'dev-other', 'test-clean',
                                      'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']):
    '''
        Loads and converts an voice dataset called "librispeech" to the DatasetType `AUDIO_FILES`.
        Refer to http://www.openslr.org/resources/12 for more details on the dataset.

        Before running this, you'll need to install SOX http://sox.sourceforge.net/
        and run `pip install sox==1.3.7`.

        :param str data_dir: Directory to save the output zip files
        :param str[] parts: Parts of the "librispeech" to load, default to the whole dataset
    '''

    print("Downloading LibriSppech dataset (55GB) into a tmp file...")

    tar_gz_urls = [part_to_url.get(part) for part in parts]

    for tar_gz_url in tar_gz_urls:
        _maybe_load_chunk(data_dir, tar_gz_url)
        

def _maybe_load_chunk(data_dir, tar_gz_url):
    filename = os.path.split(tar_gz_url)[1].split('.')[0]
    dataset_path = os.path.join(data_dir, f'{filename}.zip')

    if os.path.exists(dataset_path):
        print(f'{dataset_path} already loaded in local filesystem - skipping...')
        return

    print("You can skip the download by putting the downloaded tar.gz files into {} directory on your own..."
          .format(data_dir))
    tar_gz_path = _maybe_download(filename, data_dir, tar_gz_url)

    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    print("Extracting librivox data if not already extracted...")
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, filename), tar_gz_path)

    # Convert FLAC data to wav, from:
    #  data_dir/LibriSpeech/split/1/2/1-2-3.flac
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-3.wav
    #
    # And split LibriSpeech transcriptions, from:
    #  data_dir/LibriSpeech/split/1/2/1-2.trans.txt
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-0.txt
    #  data_dir/LibriSpeech/split-wav/1-2-1.txt
    #  data_dir/LibriSpeech/split-wav/1-2-2.txt
    #  ...
    print("Converting FLAC to WAV and splitting transcriptions...")
    work_dir = os.path.join(data_dir, LIBRIVOX_DIR)
    wav = _convert_audio_and_split_sentences(work_dir, filename, f'{filename}-wav')

    print("Writing to CSV file...")
    wav.to_csv(os.path.join(work_dir, f'{filename}-wav', "audios.csv"), index=False)

    print("Zipping required dataset format...")
    _write_dataset(os.path.join(work_dir, f'{filename}-wav'), dataset_path) 

    print('Dataset file is saved at {}'.format(dataset_path))

def _maybe_download(archive_name, target_dir, archive_url):
    # If no downloaded file is provided, download it...
    archive_path = os.path.join(target_dir, archive_name + '.tar.gz')

    if not os.path.exists(target_dir):
        print('No path "%s" - creating ...' % target_dir)
        os.makedirs(target_dir)

    if not os.path.exists(archive_path):
        print('No archive "%s" - downloading...' % archive_path)
        archive_path = download_dataset_from_url(archive_url)
    else:
        print('Found archive "%s" - not downloading.' % archive_path)
    return archive_path


def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()


def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir):
    source_dir = os.path.join(extracted_dir, data_set)
    target_dir = os.path.join(extracted_dir, dest_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop over transcription files and split each one
    #
    # The format for each file 1-2.trans.txt is:
    #  1-2-0 transcription of 1-2-0.flac
    #  1-2-1 transcription of 1-2-1.flac
    #  ...
    #
    # Each file is then split into several files:
    #  1-2-0.txt (contains transcription of 1-2-0.flac)
    #  1-2-1.txt (contains transcription of 1-2-1.flac)
    #  ...
    #
    # We also convert the corresponding FLACs to WAV in the same pass
    files = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            trans_filename = os.path.join(root, filename)
            with codecs.open(trans_filename, "r", "utf-8") as fin:
                for line in fin:
                    # Parse each segment line
                    first_space = line.find(" ")
                    seqid, transcript = line[:first_space], line[first_space+1:]

                    # We need to do the encode-decode dance here because encode
                    # returns a bytes() object on Python 3
                    transcript = unicodedata.normalize("NFKD", transcript)  \
                                            .encode("ascii", "ignore")      \
                                            .decode("ascii", "ignore")

                    transcript = transcript.lower().strip()
                    transcript = _validate_label(transcript)

                    # Convert corresponding FLAC to a WAV
                    flac_file = os.path.join(root, seqid + ".flac")
                    wav_file = os.path.join(target_dir, seqid + ".wav")

                    if not os.path.exists(wav_file) and os.path.exists(flac_file):
                        Transformer().build(flac_file, wav_file)
                    wav_filesize = os.path.getsize(wav_file)
                    files.append((seqid + ".wav", wav_filesize, transcript))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])


# Validate and normalize transcriptions. Returns a cleaned version of the label
# or None if it's invalid.
def _validate_label(label):
    # For now we can only handle [a-z ']
    if re.search(r"[0-9]|[(<\[\]&*{]", label) is not None:
        return None

    label = label.replace("-", "")
    label = label.replace("_", "")
    label = label.replace(".", "")
    label = label.replace(",", "")
    label = label.replace("?", "")
    label = label.replace("\"", "")
    label = label.strip()
    label = label.lower()

    return label if label else None

def _write_dataset(dir_path, out_dataset_path):
    # Zip and export folder as dataset
    out_path = shutil.make_archive(out_dataset_path, 'zip', dir_path)
    os.rename(out_path, out_dataset_path) # Remove additional trailing `.zip`

if __name__ == "__main__":
    load_librispeech('data/libri')
