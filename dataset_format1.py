import os
import sys
import tarfile
import h5py
from six.moves import urllib
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib import image

# http://ufldl.stanford.edu/housenumbers/train.tar.gz
# http://ufldl.stanford.edu/housenumbers/test.tar.gz
# http://ufldl.stanford.edu/housenumbers/extra.tar.gz

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


train_file = maybe_download('train.tar.gz', 404141560)
test_file = maybe_download('test.tar.gz', 276555967)
# extra_file = maybe_download('extra.tar.gz', 1955489752)

data_root = 'data_format1/'


def maybe_extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    root = data_root + root
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(data_root)
        tar.close()
    extracted_foler = root
    return extracted_foler


train_file = maybe_extract(train_file)
test_file = maybe_extract(test_file)


# img = image.imread(train_file + '/2.png')
# plt.imshow(img)
# plt.show()

def get_mat_data(f, size):
    names = f['/digitStruct/name']
    boxes = f['/digitStruct/bbox']
    data = []
    if size <= 0:
        size = names.len()
    for i in range(size):
        # for i in range(2):
        item_map = {}

        name = ''.join([chr(v[0]) for v in f[names[i][0]].value])
        item_map['filename'] = name

        def print_attrs(name, obj):
            vals = []
            if obj.shape[0] == 1:
                vals.append(obj[0][0])
            else:
                for k in range(obj.shape[0]):
                    vals.append(int(f[obj[k][0]][0][0]))
            item_map[name] = vals

        f[boxes[i][0]].visititems(print_attrs)

        data.append(item_map)
    return data


def load_dataset(size):
    with h5py.File(train_file + '/digitStruct.mat') as f:
        train_labels = get_mat_data(f, size)
    with h5py.File(test_file + '/digitStruct.mat') as f:
        test_labels = get_mat_data(f, size)

    return train_labels, test_labels


train_data, test_data = load_dataset(20)

print(train_data)
print(test_data)

index = 18
item_data = train_data[index]
image_file = item_data['filename']
label = item_data['label']
left = item_data['left']
top = item_data['top']
width = item_data['width']
height = item_data['height']
print("label = ", label, " left = ", left, " top = ", top, " width = ", width, " height = ", height)
img = image.imread(train_file + '/' + image_file)
fig, ax = plt.subplots(1)
ax.imshow(img)
for i in range(len(height)):
    rect = patches.Rectangle((left[i], top[i]), width[i], height[i], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show()
