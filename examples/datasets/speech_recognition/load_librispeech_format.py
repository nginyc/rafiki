import os

import fnmatch
import pandas
import progressbar
import tarfile
import shutil
import unicodedata

from sox import Transformer
from tensorflow.python.platform import gfile
from rafiki.model import dataset_utils
from examples.models.speech_recognition.utils.text import Alphabet, validate_label

def load(data_dir, label_filter):
    '''
        Loads and converts an voice dataset called librispeech to the DatasetType `AUDIO_FILES`.
        Refer to http://www.openslr.org/resources/12 for more details on the dataset.

        :param str data_dir: Directory to save the output zip files
    '''

    # Conditionally download data to data_dir
    print("Downloading LibriSppech data set (55GB) into a tmp file...")
    print("You can skip the download by putting the downloaded tar.gz files into {} directory on your own..."
          .format(data_dir))

    with progressbar.ProgressBar(max_value=7, widget=progressbar.AdaptiveETA) as bar:
        TRAIN_CLEAN_100_URL = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
        TRAIN_CLEAN_360_URL = "http://www.openslr.org/resources/12/train-clean-360.tar.gz"
        TRAIN_OTHER_500_URL = "http://www.openslr.org/resources/12/train-other-500.tar.gz"

        DEV_CLEAN_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
        DEV_OTHER_URL = "http://www.openslr.org/resources/12/dev-other.tar.gz"

        TEST_CLEAN_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
        TEST_OTHER_URL = "http://www.openslr.org/resources/12/test-other.tar.gz"

        def filename_of(x): return os.path.split(x)[1]
        train_clean_100 = maybe_download(filename_of(TRAIN_CLEAN_100_URL), data_dir, TRAIN_CLEAN_100_URL)
        bar.update(0)
        train_clean_360 = maybe_download(filename_of(TRAIN_CLEAN_360_URL), data_dir, TRAIN_CLEAN_360_URL)
        bar.update(1)
        train_other_500 = maybe_download(filename_of(TRAIN_OTHER_500_URL), data_dir, TRAIN_OTHER_500_URL)
        bar.update(2)

        dev_clean = maybe_download(filename_of(DEV_CLEAN_URL), data_dir, DEV_CLEAN_URL)
        bar.update(3)
        dev_other = maybe_download(filename_of(DEV_OTHER_URL), data_dir, DEV_OTHER_URL)
        bar.update(4)

        test_clean = maybe_download(filename_of(TEST_CLEAN_URL), data_dir, TEST_CLEAN_URL)
        bar.update(5)
        test_other = maybe_download(filename_of(TEST_OTHER_URL), data_dir, TEST_OTHER_URL)
        bar.update(6)


    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    print("Extracting librivox data if not already extracted...")
    with progressbar.ProgressBar(max_value=7, widget=progressbar.AdaptiveETA) as bar:
        LIBRIVOX_DIR = "LibriSpeech"
        work_dir = os.path.join(data_dir, LIBRIVOX_DIR)

        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-100"), train_clean_100)
        bar.update(0)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-360"), train_clean_360)
        bar.update(1)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-other-500"), train_other_500)
        bar.update(2)

        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
        bar.update(3)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-other"), dev_other)
        bar.update(4)

        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
        bar.update(5)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)
        bar.update(6)

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
    with progressbar.ProgressBar(max_value=7,  widget=progressbar.AdaptiveETA) as bar:
        train_100 = _convert_audio_and_split_sentences(work_dir, "train-clean-100", "train-clean-100-wav", label_filter)
        bar.update(0)
        train_360 = _convert_audio_and_split_sentences(work_dir, "train-clean-360", "train-clean-360-wav", label_filter)
        bar.update(1)
        train_500 = _convert_audio_and_split_sentences(work_dir, "train-other-500", "train-other-500-wav", label_filter)
        bar.update(2)

        dev_clean = _convert_audio_and_split_sentences(work_dir, "dev-clean", "dev-clean-wav", label_filter)
        bar.update(3)
        dev_other = _convert_audio_and_split_sentences(work_dir, "dev-other", "dev-other-wav", label_filter)
        bar.update(4)

        test_clean = _convert_audio_and_split_sentences(work_dir, "test-clean", "test-clean-wav", label_filter)
        bar.update(5)
        test_other = _convert_audio_and_split_sentences(work_dir, "test-other", "test-other-wav", label_filter)
        bar.update(6)

    # Write sets to disk as CSV files
    print("Writing CSV files...")
    train_100.to_csv(os.path.join(work_dir, "train-clean-100-wav", "audios.csv"), index=False)
    train_360.to_csv(os.path.join(work_dir, "train-clean-360-wav", "audios.csv"), index=False)
    train_500.to_csv(os.path.join(work_dir, "train-other-500-wav", "audios.csv"), index=False)

    dev_clean.to_csv(os.path.join(work_dir, "dev-clean-wav", "audios.csv"), index=False)
    dev_other.to_csv(os.path.join(work_dir, "dev-other-wav", "audios.csv"), index=False)

    test_clean.to_csv(os.path.join(work_dir, "test-clean-wav", "audios.csv"), index=False)
    test_other.to_csv(os.path.join(work_dir, "test-other-wav", "audios.csv"), index=False)

    print("Zipping required dataset formats...")
    _write_dataset(os.path.join(work_dir, "train-clean-100-wav"), os.path.join(data_dir, "train-clean-100.zip"))
    print('Train-clean-100 dataset file is saved at {}'.format(os.path.join(data_dir, "train-clean-100.zip")))

    _write_dataset(os.path.join(work_dir, "train-clean-360-wav"), os.path.join(data_dir, "train-clean-360.zip"))
    print('Train-clean-360 dataset file is saved at {}'.format(os.path.join(data_dir, "train-clean-360.zip")))

    _write_dataset(os.path.join(work_dir, "train-other-500-wav"), os.path.join(data_dir, "train-other-500.zip"))
    print('Train-other-500 dataset file is saved at {}'.format(os.path.join(data_dir, "train-other-500.zip")))

    _write_dataset(os.path.join(work_dir, "dev-clean-wav"), os.path.join(data_dir, "dev-clean.zip"))
    print('Dev-clean dataset file is saved at {}'.format(os.path.join(data_dir, "dev-clean.zip")))

    _write_dataset(os.path.join(work_dir, "dev-other-wav"), os.path.join(data_dir, "dev-other.zip"))
    print('Dev-other dataset file is saved at {}'.format(os.path.join(data_dir, "dev-other.zip")))

    _write_dataset(os.path.join(work_dir, "test-clean-wav"), os.path.join(data_dir, "test-clean.zip"))
    print('Test-clean dataset file is saved at {}'.format(os.path.join(data_dir, "test-clean.zip")))

    _write_dataset(os.path.join(work_dir, "test-other-wav"), os.path.join(data_dir, "test-other.zip"))
    print('Test-other dataset file is saved at {}'.format(os.path.join(data_dir, "test-other.zip")))

def maybe_download(archive_name, target_dir, archive_url):
    # If no downloaded file is provided, download it...
    archive_path = os.path.join(target_dir, archive_name)

    if not os.path.exists(target_dir):
        print('No path "%s" - creating ...' % target_dir)
        os.makedirs(target_dir)

    if not os.path.exists(archive_path):
        print('No archive "%s" - downloading...' % archive_path)
        archive_path = dataset_utils.download_dataset_from_uri(archive_url)
    else:
        print('Found archive "%s" - not downloading.' % archive_path)
    return archive_path


def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()


def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir, label_filter):
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
    counter = {'invalid': 0}
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
                    # returns a bytes() object on Python 3, and text_to_char_array
                    # expects a string.
                    transcript = unicodedata.normalize("NFKD", transcript)  \
                                            .encode("ascii", "ignore")      \
                                            .decode("ascii", "ignore")

                    transcript = transcript.lower().strip()

                    # Filter samples with invalid characters
                    transcript = label_filter(transcript)
                    if transcript is None:
                        counter['invalid'] += 1
                        continue

                    # Convert corresponding FLAC to a WAV
                    flac_file = os.path.join(root, seqid + ".flac")
                    wav_file = os.path.join(target_dir, seqid + ".wav")
                    if not os.path.exists(wav_file):
                        Transformer().build(flac_file, wav_file)
                    wav_filesize = os.path.getsize(wav_file)

                    files.append((seqid + ".wav", wav_filesize, transcript))

    if counter['invalid']:
        print('Warning: {} samples with invalid characters are removed! '
              'You can change the allowed set of characters in alphabet.txt file.'.format(counter['invalid']))
    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])


def _write_dataset(dir_path, out_dataset_path):
    # Zip and export folder as dataset
    out_path = shutil.make_archive(out_dataset_path, 'zip', dir_path)
    os.rename(out_path, out_dataset_path) # Remove additional trailing `.zip`

if __name__ == "__main__":
    alphabet = Alphabet('examples/datasets/speech_recognition/alphabet.txt')

    def label_filter(label):
        label = validate_label(label)
        if alphabet and label:
            try:
                [alphabet.label_from_string(c) for c in label]
            except KeyError:
                label = None
        return label

    load('data/libri', label_filter)
