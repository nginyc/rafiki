import os.path
import sys
import tarfile
from six.moves import urllib

def load(dataset_url, out_dataset_path):
    if not os.path.exists(out_dataset_path):
            os.makedirs(out_dataset_path)

    filename = dataset_url.split('/')[-1]
    filepath = os.path.join(out_dataset_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(dataset_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully download', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(out_dataset_path)

if __name__ == '__main__':
    load(
        dataset_url='http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz',
        out_dataset_path='data/imagenet'
    )