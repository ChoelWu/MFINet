import os
import csv
import h5py
import socket
import visdom
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from utils.preprocessing import preprocessing
from utils.dataset import *


class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv', append=False):
        self.filename = filename
        self.csv_file = open(filename, 'a')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        if append is False:
            for arg in args:
                writer.writerow([arg, args[arg]])
            writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)

        if append is False:
            self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


class IndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def list_all_files(rootdir):
    _files = []
    dirs = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(dirs)):
        path = os.path.join(rootdir, dirs[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


def get_visdom(port=1201):
    try:
        sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sk.settimeout(2)
        sk.connect(('localhost', port))
        sk.close
        return visdom.Visdom(port=port)
    except socket.error:
        return None


def mkdir_p(path):
    """
    make dir if not exist
    :param path:
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def group_images(data, per_row):
    """
    group a set of images row per columns
    :param data:
    :param per_row:
    :return:
    """
    assert data.shape[0] % per_row == 0
    assert data.shape[1] == 1 or data.shape[1] == 3
    data = np.transpose(data, (0, 2, 3, 1))  # corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg


def visualize(data, filename):
    """
    visualize image as PIL image
    :param data:
    :param filename:
    :return:
    """
    assert (len(data.shape) == 3)
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename)
    return img


def pred_to_imgs(pred, patch_size):
    """
    获取预测结果，以 1 为准
    :param pred:
    :param patch_size:
    :return:
    """
    assert len(pred.shape) == 3  # (Npatches, height * width, 2)
    assert pred.shape[2] == 2
    pred_images = pred[:, :, 1]
    pred_images = np.reshape(pred_images, (pred_images.shape[0], 1, patch_size, patch_size))
    return pred_images


def get_training_patchs(train_imgs, train_gts, patch_size, patch_num, inside_FOV):
    """
    Load the original data and return the extracted patches for training/testing
    :param train_imgs:
    :param train_gts:
    :param patch_size:
    :param patch_num:
    :param inside_FOV:
    :return:
    """
    data_consistency_check(train_imgs, train_gts)
    train_imgs_patches, train_masks_patches = extract_random(train_imgs, train_gts, patch_size, patch_num, inside_FOV)
    data_consistency_check(train_imgs_patches, train_masks_patches)
    print("train PATCHES images shape: " + str(train_imgs_patches.shape))
    print("train PATCHES masks shape: " + str(train_masks_patches.shape))
    print("train PATCHES images range (min-max): " + str(np.min(train_imgs_patches)) + ' - ' + str(np.max(train_imgs_patches)))
    return train_imgs_patches, train_masks_patches


def get_testing_patchs(test_imgs, test_gts, patch_size):
    """
    Load the original data and return the extracted patches for training/testing
    :param test_imgs:
    :param test_gts:
    :param patch_size
    :return:
    """
    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = paint_border(test_imgs, patch_size)
    test_gts = paint_border(test_gts, patch_size)

    data_consistency_check(test_imgs, test_gts)

    print("test images shape:" + str(test_imgs.shape))
    print("test masks shape:" + str(test_gts.shape))
    print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))

    # extract the test patches from the full images
    test_imgs_patches = extract_ordered(test_imgs, patch_size)
    test_masks_patches = extract_ordered(test_gts, patch_size)

    data_consistency_check(test_imgs_patches, test_masks_patches)

    print("test PATCHES images shape:" + str(test_imgs_patches.shape))
    print("test PATCHES masks shape:" + str(test_masks_patches.shape))
    print("test PATCHES images range (min-max): " + str(np.min(test_masks_patches)) + ' - ' + str(np.max(test_masks_patches)))

    return test_imgs, test_gts, test_imgs_patches, test_masks_patches


def get_data_testing_overlap(test_imgs, test_gts, patch_size, stride_size):
    """
    Load the original data and return the extracted patches for testing
    return the ground truth in its original shape
    :param test_imgs:
    :param test_gts:
    :param patch_size:
    :param stride_size:
    :return:
    """
    # extend both images and masks so they can be divided exactly by the patches dimensions
    new_test_imgs = paint_border_overlap(test_imgs, patch_size, stride_size)

    # check masks are within 0-1
    assert (np.max(test_gts) == 1 and np.min(test_gts) == 0)

    print("test images shape: " + str(new_test_imgs.shape))
    print("test mask shape: " + str(test_gts.shape))
    print("test images range (min-max): " + str(np.min(new_test_imgs)) + ' - ' + str(np.max(new_test_imgs)))

    # extract the TEST patches from the full images
    test_imgs_patches = extract_ordered_overlap(new_test_imgs, patch_size, stride_size)

    print("test PATCHES images shape: " + str(test_imgs_patches.shape))
    print("test PATCHES images range (min-max): " + str(np.min(test_imgs_patches)) + ' - ' + str(np.max(test_imgs_patches)))

    return test_imgs_patches, new_test_imgs.shape[2], new_test_imgs.shape[3]


def data_consistency_check(imgs, masks):
    """
    data consinstency check
    :param imgs:
    :param masks:
    :return:
    """
    assert (len(imgs.shape) == len(masks.shape))
    assert (imgs.shape[0] == masks.shape[0])
    assert (imgs.shape[2] == masks.shape[2])
    assert (imgs.shape[3] == masks.shape[3])
    assert (masks.shape[1] == 1)
    assert (imgs.shape[1] == 1 or imgs.shape[1] == 3)


def extract_random(full_imgs, full_masks, patch_size, N_patches, inside=True):
    """
    extract patches randomly in the full images
    :param full_imgs:
    :param full_masks:
    :param patch_size:
    :param N_patches:
    :param inside:
    :return:
    """
    if N_patches % full_imgs.shape[0] != 0:
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    assert (full_masks.shape[1] == 1)  # masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    patches = np.empty((N_patches, full_imgs.shape[1], patch_size, patch_size))
    patches_masks = np.empty((N_patches, full_masks.shape[1], patch_size, patch_size))
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  # N_patches equally divided in the full images
    # print("patches per full image: " + str(patch_per_img))
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        k = 0
        while k < patch_per_img:
            x = random.randint(0 + int(patch_size / 2), img_w - int(patch_size / 2))
            y = random.randint(0 + int(patch_size / 2), img_h - int(patch_size / 2))
            if inside is True:
                if is_patch_inside_FOV(x, y, img_w, img_h, patch_size) is False:
                    continue

            patches[iter_tot] = full_imgs[i, :, y - patch_size // 2:y + patch_size // 2, x - patch_size // 2:x + patch_size // 2]
            patches_masks[iter_tot] = full_masks[i, :, y - patch_size // 2:y + patch_size // 2, x - patch_size // 2:x + patch_size // 2]
            iter_tot += 1  # total
            k += 1  # per full_img
    return patches, patches_masks


def is_patch_inside_FOV(x, y, img_w, img_h, patch_size):
    """
    check if the patch is fully contained in the FOV
    :param x:
    :param y:
    :param img_w:
    :param img_h:
    :param patch_size:
    :return:
    """
    x_ = x - int(img_w / 2)  # origin (0,0) shifted to image center
    y_ = y - int(img_h / 2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_size * np.sqrt(2.0) / 2.0)
    # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_ * x_) + (y_ * y_))
    if radius < R_inside:
        return True
    else:
        return False


def extract_ordered(full_imgs, patch_size):
    """
    Divide all the full_imgs in pacthes
    :param full_imgs:
    :param patch_size
    :return:
    """
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    N_patches_h = int(img_h / patch_size)  # round to lowest int
    if img_h % patch_size != 0:
        print("warning: " + str(N_patches_h) + " patches in height, with about " + str(img_h % patch_size) + " pixels left over")
    N_patches_w = int(img_w / patch_size)  # round to lowest int
    if img_h % patch_size != 0:
        print("warning: " + str(N_patches_w) + " patches in width, with about " + str(img_w % patch_size) + " pixels left over")
    print("number of patches per image: " + str(N_patches_h * N_patches_w))
    N_patches_tot = (N_patches_h * N_patches_w) * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_size, patch_size))

    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patches[iter_tot] = full_imgs[i, :, h * patch_size:(h * patch_size) + patch_size, w * patch_size:(w * patch_size) + patch_size]
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches  # array with all the full_imgs divided in patches


def paint_border_overlap(full_imgs, patch_size, stride_size):
    """
    给图像加右侧和下侧加几个像素
    :param full_imgs:
    :param patch_size:
    :param stride_size:
    :return:
    """
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    leftover_h = (img_h - patch_size) % stride_size  # leftover on the h dim
    leftover_w = (img_w - patch_size) % stride_size  # leftover on the w dim
    if leftover_h != 0:  # change dimension of img_h
        # print("the side H is not compatible with the selected stride of " + str(stride_size))
        # print("img_h " + str(img_h) + ", patch_h " + str(patch_size) + ", stride_h " + str(stride_size))
        # print("(img_h - patch_h) MOD stride_h: " + str(leftover_h))
        # print("So the H dim will be padded with additional " + str(stride_size - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h + (stride_size - leftover_h), img_w))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if leftover_w != 0:  # change dimension of img_w
        # print("the side W is not compatible with the selected stride of " + str(stride_size))
        # print("img_w " + str(img_w) + ", patch_w " + str(patch_size) + ", stride_w " + str(stride_size))
        # print("(img_w - patch_w) MOD stride_w: " + str(leftover_w))
        # print("So the W dim will be padded with additional " + str(stride_size - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w + (stride_size - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2], 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    # print("new full images shape: " + str(full_imgs.shape))  # 20, 1, 588, 568
    return full_imgs


def extract_ordered_overlap(full_imgs, patch_size, stride_size):
    """
    Divide all the full_imgs in pacthes
    :param full_imgs:
    :param patch_size:
    :param stride_size:
    :return:
    """
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    assert ((img_h - patch_size) % stride_size == 0 and (img_w - patch_size) % stride_size == 0)
    N_patches_img = ((img_h - patch_size) // stride_size + 1) * ((img_w - patch_size) // stride_size + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]
    # print("Number of patches on h : " + str(((img_h - patch_size) // stride_size + 1)))
    # print("Number of patches on w : " + str(((img_w - patch_size) // stride_size + 1)))
    # print("number of patches per image: " + str(N_patches_img) + ", totally for this dataset: " + str(N_patches_tot))
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_size, patch_size))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_size) // stride_size + 1):
            for w in range((img_w - patch_size) // stride_size + 1):
                patch = full_imgs[i, :, h * stride_size:(h * stride_size) + patch_size, w * stride_size:(w * stride_size) + patch_size]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches  # array with all the full_imgs divided in patches


def recompone_overlap(preds, img_h, img_w, stride_size):
    """
    patches ==> pred ground truth
    :param preds: 预测patch
    :param img_h: 新的图像高度，由于裁剪时进行了部分填充，所以和原图像不一样大小
    :param img_w: 新的图像宽度，由于裁剪时进行了部分填充，所以和原图像不一样大小
    :param stride_h: 裁剪时的步长
    :param stride_w:  裁剪时的步长
    :return:
    """
    assert (len(preds.shape) == 4)
    assert (preds.shape[1] == 1 or preds.shape[1] == 3)  # check the channel is 1 or 3
    patch_size = preds.shape[2]
    N_patches_h = (img_h - patch_size) // stride_size + 1
    N_patches_w = (img_w - patch_size) // stride_size + 1
    N_patches_img = N_patches_h * N_patches_w
    #print("N_patches_h: " + str(N_patches_h))
    #print("N_patches_w: " + str(N_patches_w))
    #print("N_patches_img: " + str(N_patches_img))
    assert (preds.shape[0] % N_patches_img == 0)
    N_full_imgs = preds.shape[0] // N_patches_img
    # print("According to the dimension inserted, there are " + str(N_full_imgs) + " full images (of " + str(img_h) + "x" + str(img_w) + " each)")
    full_prob = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_size) // stride_size + 1):
            for w in range((img_w - patch_size) // stride_size + 1):
                full_prob[i, :, h * stride_size:(h * stride_size) + patch_size, w * stride_size:(w * stride_size) + patch_size] += preds[k]
                full_sum[i, :, h * stride_size:(h * stride_size) + patch_size, w * stride_size:(w * stride_size) + patch_size] += 1
                k += 1

    return full_prob / full_sum


def recompone(data, N_h, N_w):
    """
    Recompone the full images with the patches
    :param data:
    :param N_h:
    :param N_w:
    :return:
    """
    assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    assert (len(data.shape) == 4)
    N_pacth_per_img = N_w * N_h
    assert (data.shape[0] % N_pacth_per_img == 0)
    N_full_imgs = data.shape[0] / N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    # define and start full recompone
    full_recomp = np.empty((int(N_full_imgs), data.shape[1], int(N_h * patch_h), int(N_w * patch_w)))
    k = 0  # iter full img
    s = 0  # iter single patch
    while s < data.shape[0]:
        single_recon = np.empty((data.shape[1], N_h * patch_h, N_w * patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w] = data[s]
                s += 1
        full_recomp[k] = single_recon
        k += 1
    assert (k == N_full_imgs)
    return full_recomp


def paint_border(data, patch_size):
    """
    Extend the full images becasue patch divison is not exact
    :param data:
    :param patch_size:
    :return:
    """
    assert (len(data.shape) == 4)  # 4D arrays
    assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    img_h = data.shape[2]
    img_w = data.shape[3]
    if img_h % patch_size == 0:
        new_img_h = img_h
    else:
        new_img_h = (img_h // patch_size + 1) * patch_size
    if (img_w % patch_size) == 0:
        new_img_w = patch_size
    else:
        new_img_w = (img_w // patch_size + 1) * patch_size
    new_data = np.zeros((data.shape[0], data.shape[1], new_img_h, new_img_w))
    new_data[:, :, 0:img_h, 0:img_w] = data[:, :, :, :]
    return new_data


def pred_only_FOV(data_imgs, data_masks, original_imgs_border_masks):
    """
    return only the pixels contained in the FOV, for both images and masks
    :param data_imgs:
    :param data_masks:
    :param original_imgs_border_masks:
    :return:
    """
    assert (len(data_imgs.shape) == 4 and len(data_masks.shape) == 4)  # 4D arrays
    assert (data_imgs.shape[0] == data_masks.shape[0])
    assert (data_imgs.shape[2] == data_masks.shape[2])
    assert (data_imgs.shape[3] == data_masks.shape[3])
    assert (data_imgs.shape[1] == 1 and data_masks.shape[1] == 1)  # check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  # loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV(x, y, original_imgs_border_masks[i]) is True:
                    new_pred_imgs.append(data_imgs[i, :, y, x])
                    new_pred_masks.append(data_masks[i, :, y, x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks


def kill_border(data, masks):
    """
    function to set to black everything outside the FOV, in a full image
    :param data:
    :param original_imgs_border_masks:
    :return:
    """
    assert (len(data.shape) == 4)  # 4D arrays
    assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):  # loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV(x, y, masks[i]) is False:
                    data[i, :, y, x] = 0.0


def inside_FOV(x, y, masks):
    """
    判断点(x,y)是否在mask大于0的部分，通过mask文件来固定非眼底值为0
    :param i:
    :param x:
    :param y:
    :param DRIVE_masks:
    :return:
    """
    assert (len(masks.shape) == 2 or len(masks.shape) == 3)
    if x >= masks.shape[2] or y >= masks.shape[1]:
        return False

    if masks[0:, y, x] > 0:
        return True
    else:
        return False


def get_orig_datasets(args):
    data_path = args['data_path'] + args['dataset'] + '/'
    train_imgs_original = load_hdf5(data_path + 'imgs_train.hdf5')
    train_imgs = preprocessing(train_imgs_original)
    train_gts = load_hdf5(data_path + 'ground_truth_train.hdf5')
    train_gts = train_gts / 255.
    train_masks = load_hdf5(data_path + 'border_masks_train.hdf5')

    test_imgs_original = load_hdf5(data_path + 'imgs_test.hdf5')
    test_imgs = preprocessing(test_imgs_original)
    test_gts = load_hdf5(data_path + 'ground_truth_test.hdf5')
    test_gts = test_gts / 255.
    test_masks = load_hdf5(data_path + 'border_masks_test.hdf5')

    return train_imgs, train_gts, train_masks, test_imgs, test_gts, test_masks


def get_orig_SATRE_datasets(args):
    data_path = args['data_path'] + args['dataset'] + '/'
    imgs_original = load_hdf5(data_path + 'imgs.hdf5')
    imgs = preprocessing(imgs_original)
    gts = load_hdf5(data_path + 'ground_truth.hdf5')
    gts = gts / 255.
    masks = load_hdf5(data_path + 'border_masks.hdf5')

    train_imgs = np.delete(imgs, args['fold'], axis=0)
    train_gts = np.delete(gts, args['fold'], axis=0)
    train_masks = np.delete(masks, args['fold'], axis=0)
    
    test_imgs = imgs[args['fold']].reshape((1, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    test_gts = gts[args['fold']].reshape((1, gts.shape[1], gts.shape[2], gts.shape[3]))
    test_masks = masks[args['fold']].reshape((1, masks.shape[1], masks.shape[2], masks.shape[3]))
    
    return train_imgs, train_gts, train_masks, test_imgs, test_gts, test_masks


def get_orig_SATRE_datasets_for_test(args):
    data_path = args['data_path'] + args['dataset'] + '/'
    imgs_original = load_hdf5(data_path + 'imgs.hdf5')
    imgs = preprocessing(imgs_original)
    gts = load_hdf5(data_path + 'ground_truth.hdf5')
    gts = gts / 255.
    masks = load_hdf5(data_path + 'border_masks.hdf5')
    
    test_imgs = imgs[args['fold']].reshape((1, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    test_gts = gts[args['fold']].reshape((1, gts.shape[1], gts.shape[2], gts.shape[3]))
    test_masks = masks[args['fold']].reshape((1, masks.shape[1], masks.shape[2], masks.shape[3]))
    
    test_imgs_patches, new_height, new_width = get_data_testing_overlap(
        test_imgs=test_imgs,
        test_gts=test_gts,
        patch_size=args['patch_size'],
        stride_size=args['stride_size'],
    )
    
    return test_imgs, test_gts, test_masks, test_imgs_patches, new_height, new_width


def get_dataset_for_test(args):
    data_path = args['data_path'] + args['dataset'] + '/'
    test_imgs_original = load_hdf5(data_path + 'imgs_test.hdf5')
    test_imgs = preprocessing(test_imgs_original)
    test_gts = load_hdf5(data_path + 'ground_truth_test.hdf5')
    test_gts = test_gts / 255.
    test_masks = load_hdf5(data_path + 'border_masks_test.hdf5')

    test_imgs_patches, new_height, new_width = get_data_testing_overlap(
        test_imgs=test_imgs,
        test_gts=test_gts,
        patch_size=args['patch_size'],
        stride_size=args['stride_size'],
    )
    return test_imgs, test_gts, test_masks, test_imgs_patches, new_height, new_width


if __name__ == '__main__':
    # test_masks = load_hdf5('/home/izhangh/work/python/N2UNet/datasets/DRIVE/test/DRIVE_dataset_borderMasks_test.hdf5')
    # print(test_masks.shape)
    data_path = '/home/izhangh/work/python/N2UNet/datasets/STARE/'

    train_imgs_original = load_hdf5(data_path + 'imgs_train.hdf5')
    train_imgs = preprocessing(train_imgs_original)
    train_gts = load_hdf5(data_path + 'ground_truth_train.hdf5')
    train_gts = train_gts / 255.
