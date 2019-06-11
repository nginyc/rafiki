import sys
import os

import pandas
import shutil
from rafiki.model import dataset_utils


def load(data_dir):
    '''
        Loads and converts an sample voice from dataset LDC93S1 to the DatasetType `AUDIO_FILES`.
        This file only serves a demonstrative purpose since the full LDC93S1 corpus is not free.

        :param str data_dir: Directory to save the output zip files
    '''

    # Conditionally download data
    LDC93S1_DIR = "ldc93s1"
    work_dir = os.path.join(data_dir, LDC93S1_DIR)

    LDC93S1_BASE = "LDC93S1"
    LDC93S1_BASE_URL = "https://catalog.ldc.upenn.edu/desc/addenda/"
    audio_file = maybe_download(LDC93S1_BASE + ".wav", work_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".wav")
    trans_file = maybe_download(LDC93S1_BASE + ".txt", work_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")
    shutil.copyfile(audio_file, os.path.join(work_dir, LDC93S1_BASE + ".wav"))
    with open(trans_file, "r") as fin:
        transcript = ' '.join(fin.read().strip().lower().split(' ')[2:]).replace('.', '')

    df = pandas.DataFrame(data=[(LDC93S1_BASE + ".wav", os.path.getsize(audio_file), transcript)],
                          columns=["wav_filename", "wav_filesize", "transcript"])
    df.to_csv(os.path.join(work_dir, "audios.csv"), index=False)

    print("Zipping required dataset formats...")
    _write_dataset(work_dir, os.path.join(data_dir, "ldc93s1.zip"))
    print('LDC93S1 dataset file is saved at {}'.format(os.path.join(data_dir, "ldc93s1.zip")))


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


def _write_dataset(dir_path, out_dataset_path):
    # Zip and export folder as dataset
    out_path = shutil.make_archive(out_dataset_path, 'zip', dir_path)
    os.rename(out_path, out_dataset_path) # Remove additional trailing `.zip`

if __name__ == "__main__":
    load('data/ldc93s1')
