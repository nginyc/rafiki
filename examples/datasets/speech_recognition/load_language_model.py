import os
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

# Grab trie
trie_url = 'https://github.com/ZhaoxuanWu/lm_files/raw/master/trie'
trie_filename = os.path.join(os.path.abspath('data'), 'trie')
request.urlretrieve(trie_url, trie_filename, reporthook)
print('The trie file is saved at {}'.format(trie_filename))

# Grab lm.binary
lm_binary_url = 'https://github.com/ZhaoxuanWu/lm_files/raw/master/lm.binary'
lm_binary_filename = os.path.join(os.path.abspath('data'), 'lm.binary')
request.urlretrieve(lm_binary_url, lm_binary_filename, reporthook)
print('The language model binary file is saved at {}'.format(lm_binary_filename))
