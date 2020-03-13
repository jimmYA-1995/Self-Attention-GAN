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


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    h, w = x.shape[:2]
    crop_h = min(h, w) 

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
    img = cv2.cvtColor(cv2.imread(image_path) ,cv2.COLOR_BGR2RGB)
    out = transform(img, image_size, is_crop, resize_w)
    return out

def colorize(img):
    # convert grayscale to rgb by concatenation
    if img.ndim == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.concatenate([img, img, img], axis=2)
    
    # throw up Alpha in RGBA images
    if img.shape[2] == 4:
        img = img[:, :, 0:3]
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(args):
    metadata = {'img_size': args.img_size}
    files = []

    if args.dataset == "imagenet":
        dirs = glob(os.path.join(args.path,"train/n*"))
        assert len(dirs) == 1000, len(dirs)
        shuffle(dirs)
        dirname = dirs[:] if args.n_class == -1 else dirs[:args.n_class]
        for d in dirname:
            files.extend(glob(d+"/*JPEG"))
        
        dirs = [d.split('/')[-1] for d in dirname]
        dirs = sorted(dirs)
        str_to_int = dict(zip(dirs, range(len(dirs))))
        metadata.update({'num_classes': len(dirs)})
        target_path = 'imagenet_train_labeled_{}_{}'.format(args.img_size, args.n_class)
    elif "Lsun" in args.dataset:
        pattern = os.path.join(args.path, "data/*.jpg") 
        files = glob(pattern)
        metadata.update({'num_classes': 1})
        target_path = '{}_unlabeled_{}'.format(args.dataset, args.img_size)
    else:
        raise ValueError("only support imagenet and Lsun series")
    
    if not os.path.exists(os.path.join(args.path, target_path)):
        os.makedirs(os.path.join(args.path, target_path))
        
    assert len(files) > 0
    metadata.update({'num_records': len(files)})
    shuffle(files)

    with open(os.path.join(args.path, target_path, 'metadata.pickle'), 'wb') as f:
        pickle.dump(metadata, f)

    outfile = os.path.join(args.path, target_path, 'data.tfrecords')
    writer = tf.io.TFRecordWriter(outfile)

    for f in tqdm(files):
        image = get_image(f, args.img_size, is_crop=True, resize_w=args.img_size)
        image = colorize(image)
        assert image.shape == (args.img_size, args.img_size, 3)
        image += 1.
        image *= (255. / 2.)
        image = image.astype('uint8')
        image_raw = image.tostring()
        
        if args.dataset == 'imagenet':
            class_str = f.split('/')[-2]
            label = str_to_int[class_str]
        elif "Lsun" in args.dataset:
            label = tf.zeros(shape=(), dtype=tf.int64)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    #'height': _int64_feature(args.img_size),
                    #'width': _int64_feature(args.img_size),
                    #'depth': _int64_feature(3),
                    'image_raw': _bytes_feature(image_raw),
                    'label': _int64_feature(label)
                }))
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str,
                        help='root path of dataset')
    parser.add_argument('--dataset', type=str, default='imagenet', help='dataset.')
    parser.add_argument('--img_size', type=int, default='64',
                        help='target image size')
    parser.add_argument('--n_class', type=int, default=-1,
                        help='num of classes')
    args = parser.parse_args()
    main(args)
