import gzip
import io
import sys
import time
from urllib import request


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


# Grab corpus.
url = 'http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz'
data_upper = '/tmp/upper.txt.gz'
request.urlretrieve(url, data_upper, reporthook)
print('Finished downloading, starting cleanup \n')

# Convert to lowercase and cleanup.
data_lower = '/tmp/lower.txt'
with open(data_lower, 'w', encoding='utf-8') as lower:
    with io.TextIOWrapper(io.BufferedReader(gzip.open(data_upper)), encoding='utf8') as upper:
        for line in upper:
            lower.write(line.lower())

print('Finished converting case, saved to /tmp/ folder \n')