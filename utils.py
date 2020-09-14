import os
import cv2
import time, datetime
import random
import numpy as np
import matplotlib.pyplot as plt

def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

def read_images(dataset_path, mode):
    imagepaths, labels = list(), list()
    if mode == 'file':
        # Read dataset file
        with open(dataset_path) as f:
            data = f.read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # List the directory
        try:  # Python 2
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:  # Python 3
            classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            except Exception:  # Python 3
                walk = os.walk(c_dir).__next__()
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.jpg') or sample.endswith('.jpeg') or sample.endswith('.bmp') or sample.endswith('.png'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append([label])
            label += 1
    else:
        raise Exception("Unknown mode.")

    images = list()
    for p in imagepaths:
        image = cv2.imread(p, 0)
        if image is  None:
            continue
        image = cv2.resize(image, (256, 256))
        image = np.float32(image)/127.5 - 1
        image = np.expand_dims(image, axis=2)
        images.append(image)
        #cv2.imshow("image", image)

    images = np.array(images)
    labels = np.array(labels, dtype=np.int32)

    print("images: %d" % len(images))
    print("labels: %d" % len(labels))

    if len(images) != len(labels):
        print("Error---------------------")

    else:
        return images, labels

def random_crop(image, crop_height, crop_width):
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        return image[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))

def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

def interpolate_points(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)

def plot_generated(sample_data, n):
    plt.figure(figsize=(10, 10))
    for i in range(n * n):
        plt.subplot(n, n, 1+i)
        plt.axis('off')
        plt.imshow((sample_data[i]).reshape([256, 256]), cmap='gray')
    plt.savefig('fig.png', dpi=300)
    plt.show()

def save_plot_generated(sample_data, n, str):
    plt.figure(figsize=(10, 10))
    for i in range(n * n):
        plt.subplot(n, n, 1+i)
        plt.axis('off')
        plt.imshow((sample_data[i]).reshape([256, 256]), cmap='gray')
    plt.savefig(str, dpi=300)