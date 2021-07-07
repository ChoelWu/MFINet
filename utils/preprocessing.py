import cv2
import numpy as np


def preprocessing(data):
    """
    Image preprocess for both train set and test set
    :param data:
    :return:
    """
    assert (len(data.shape) == 4)
    assert (data.shape[1] == 3)
    imgs = rgb2gray(data)
    imgs = dataset_normalized(imgs)
    imgs = clahe_equalized(imgs)
    imgs = adjust_gamma(imgs, 1.2)
    imgs = imgs / 255.
    return imgs


def preprocessing_step(imgs, args):
    """
    Image preprocess for both train set and test set
    :param data:
    :return:
    """
    assert (len(imgs.shape) == 4)
    assert (imgs.shape[1] == 3)
    if args['pre_rgb']:
        imgs = rgb2gray(imgs)
    if args['pre_norm']:
        imgs = dataset_normalized(imgs)
    if args['pre_clahe']:
        imgs = clahe_equalized(imgs)
    if args['pre_gamma']:
        imgs = adjust_gamma(imgs, 1.2)
    imgs = imgs / 255.
    return imgs


def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:, 0, :, :] * 0.299 + rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs


def get_green(rgb):
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:, 0, :, :] * 0. + rgb[:, 1, :, :] * 1. + rgb[:, 2, :, :] * 0.
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs


def dataset_normalized(imgs):
    """
    normalize over the dataset
    :param imgs:
    :return:
    """
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


def clahe_equalized(imgs):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization.
    After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
    :param imgs:
    :return:
    """
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs


def histo_equalized(imgs):
    """
    histogram equalization
    :param imgs:
    :return:
    """
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = cv2.equalizeHist(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized
