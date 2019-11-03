import os
import time
import cv2
import numpy as np
import tensorflow as tf
import pickle
import argparse

from glob import glob
from tqdm import tqdm 
from random import shuffle

# this is based on tensorflow tutorial code
# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# TODO: it is probably very wasteful to store these images as raw numpy
# strings, because that is not compressed at all.
# i am only doing that because it is what the tensorflow tutorial does.
# should probably figure out how to store them as JPEG.

IMSIZE = 128

def center_crop(x, crop_h, crop_w=None, resize_w=64):

    h, w = x.shape[:2]
    crop_h = min(h, w)  # we changed this to override the original DCGAN-TensorFlow behavior
                        # Just use as much of the image as possible while keeping it square

    if crop_w is None:
        crop_w = crop_h
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(x[j:j+crop_h, i:i+crop_w],
                               (resize_w, resize_w), interpolation = cv2.INTER_AREA)

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    cropped_image = center_crop(image, npx, resize_w=resize_w)
    return np.array(cropped_image)/127.5 - 1.

def get_image(image_path, image_size, is_crop=True, resize_w=64):
    global index
    out = transform(cv2.imread(image_path), image_size, is_crop, resize_w)
    return out

def colorize(img):
    if img.ndim == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.concatenate([img, img, img], axis=2)
    if img.shape[2] == 4:
        img = img[:, :, 0:3]
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(args):

    pattern = os.path.join(args.path,"train/n*/*JPEG")
    files = glob(pattern)
    assert len(files) > 0
    # assert len(files) > 1000000, len(files) #small: 19439
    shuffle(files)

    dirs = glob(os.path.join(args.path,"train/n*"))
    # assert len(dirs) == 1000, len(dirs) # small: 774
    dirs = [d.split('/')[-1] for d in dirs]
    dirs = sorted(dirs)
    str_to_int = dict(zip(dirs, range(len(dirs))))

    with open(os.path.join(args.path, 'metadata.pickle'), 'wb') as f:
        pickle.dump({
            'num_records': len(files),
            'num_classes': len(dirs)
        }, f)

    outfile = os.path.join(args.path, 'imagenet_train_labeled_{}.tfrecords'.format(str(IMSIZE)))
    writer = tf.io.TFRecordWriter(outfile)

    for f in tqdm(files):
        image = get_image(f, IMSIZE, is_crop=True, resize_w=IMSIZE)
        image = colorize(image)
        assert image.shape == (IMSIZE, IMSIZE, 3)
        image += 1.
        image *= (255. / 2.)
        image = image.astype('uint8')
        
        image_raw = image.tostring()
        class_str = f.split('/')[-2]
        label = str_to_int[class_str]
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IMSIZE),
            'width': _int64_feature(IMSIZE),
            'depth': _int64_feature(3),
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str,
                        help='root path of dataset')
    args = parser.parse_args()
    main(args)
