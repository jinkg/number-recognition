import os
import sys
import scipy.io as sio
import numpy as np
from six.moves import urllib

# http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
# http://ufldl.stanford.edu/housenumbers/test_32x32.mat
# http://ufldl.stanford.edu/housenumbers/train_32x32.mat

url = 'http://ufldl.stanford.edu/housenumbers/'

last_percent_reported = None


def download_progress_hook(count, block_size, total_size):
    """A hook to report the progress of a download. This is mostly intended for users with
      slow internet connections. Reports every 5% change in download progress.
      """
    global last_percent_reported
    percent = int(count * block_size * 100 / total_size)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found an verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


train_file = maybe_download('data_format2/train_32x32.mat', 182040794)
test_file = maybe_download('data_format2/test_32x32.mat', 64275384)


def get_data(filename):
    data = sio.loadmat(filename)
    x = np.array(data['X']).transpose((3, 0, 1, 2))
    y = data['y']
    return x, y


def get_train_data():
    return get_data(train_file)


def get_test_data():
    return get_data(test_file)
