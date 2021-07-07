import os
import h5py
import numpy as np
from PIL import Image
from skimage.measure import label
#import matplotlib.pyplot as plt
from keras.preprocessing import image


def write_hdf5(arr, outfile):
    with h5py.File(outfile, 'w') as f:
        f.create_dataset('image', data=arr, dtype=arr.dtype)


def get_HRF():
    dataset_path = './datasets/HRF/'
    imgs_path = './datasets/HRF/images/'
    ground_truth_path = './datasets/HRF/manual1/'
    border_masks_path = './datasets/HRF/mask/'

    imgs = []
    ground_truth = []
    border_masks = []
    files = np.array(sorted(list(os.listdir(imgs_path))))
    for file in files:
        # original
        print('original image: ' + file)
        img = np.asarray(image.load_img(imgs_path + file))
        imgs.append(img)
        #plt.imshow(img)
        #plt.show()
        # corresponding ground truth
        ground_truth_name = file[:-4] + '.tif'
        print('ground truth name: ' + ground_truth_name)
        g_truth = np.array(image.load_img(ground_truth_path + ground_truth_name, grayscale=True))
        #plt.imshow(g_truth, cmap='gray')
        #plt.show()
        ground_truth.append(g_truth)
        # corresponding border masks
        border_masks_name = file[:-4] + '_mask.tif'
        print('border masks name: ' + border_masks_name)
        b_mask = np.array(image.load_img(border_masks_path + border_masks_name, grayscale=True))
        #plt.imshow(b_mask, cmap='gray')
        #plt.show()
        border_masks.append(b_mask)

    print('imgs max: ' + str(np.max(imgs)) + ', min: ' + str(np.min(imgs)))
    assert (np.max(ground_truth) == 255 and np.max(border_masks) == 255)
    assert (np.min(ground_truth) == 0 and np.min(border_masks) == 0)
    print('ground truth and border masks are correctly withih pixel value range 0-255 (black-white)')
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    ground_truth = np.array(ground_truth)
    ground_truth = ground_truth.reshape(ground_truth.shape[0], 1, ground_truth.shape[1], ground_truth.shape[2])
    border_masks = np.array(border_masks)
    border_masks = border_masks.reshape(border_masks.shape[0], 1, border_masks.shape[1], border_masks.shape[2])
    print('imgs shape: ' + str(imgs.shape))
    print('ground_truth shape: ' + str(ground_truth.shape))
    print('border_masks shape: ' + str(border_masks.shape))

    print('saving train datasets')
    write_hdf5(imgs[0:15], dataset_path + 'imgs_train.hdf5')
    write_hdf5(ground_truth[0:15], dataset_path + 'ground_truth_train.hdf5')
    write_hdf5(border_masks[0:15], dataset_path + 'border_masks_train.hdf5')
    print('saving test datasets')
    write_hdf5(imgs[15:], dataset_path + 'imgs_test.hdf5')
    write_hdf5(ground_truth[15:], dataset_path + 'ground_truth_test.hdf5')
    write_hdf5(border_masks[15:], dataset_path + 'border_masks_test.hdf5')


def get_CHASE():
    images_path = "./datasets/CHASE/images/"
    imgs = []
    ground_truth = []
    masks = []
    for i in range(1, 15):
        i_str = str(i)
        if i < 10: i_str = '0' + i_str

        for j in ['L', 'R']:
            # original
            file_name = 'Image_' + i_str + j
            print("original image: " + file_name + '.jpg')
            img = np.asarray(image.load_img(images_path + file_name + '.jpg'))
            imgs.append(img)
            plt.imshow(img)
            plt.show()

            # corresponding ground truth
            print("ground truth name: " + file_name + '_1stHO.png')
            g_truth = np.asarray(image.load_img(images_path + file_name + '_1stHO.png'))[:, :, 0]
            # g_truth = label(np.asarray(g_truth))
            plt.imshow(g_truth, cmap='gray')
            plt.show()
            ground_truth.append(g_truth)

            # masks
            print("original image: " + file_name + '_mask.jpg')
            mask = np.asarray(image.load_img(images_path + file_name + '_mask.jpg'))[:, :, 0]
            masks.append(mask)
            plt.imshow(mask, cmap='gray')
            plt.show()

    print("imgs max: " + str(np.max(imgs)))
    print("imgs min: " + str(np.min(imgs)))
    assert (np.max(ground_truth) == 255 and np.min(ground_truth) == 0)
    print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
    # reshaping for my standard tensors
    imgs = np.transpose(np.array(imgs), (0, 3, 1, 2))
    ground_truth = np.array(ground_truth)
    ground_truth = ground_truth.reshape(ground_truth.shape[0], 1, ground_truth.shape[1], ground_truth.shape[2])
    masks = np.array(masks)
    masks = masks.reshape(masks.shape[0], 1, masks.shape[1], masks.shape[2])
    print('imgs shape: ' + str(imgs.shape))
    print('ground_truth shape: ' + str(ground_truth.shape))
    print('masks shape: ' + str(masks.shape))

    write_hdf5(imgs[0:20], images_path + "../imgs_train.hdf5")
    write_hdf5(ground_truth[0:20], images_path + "../ground_truth_train.hdf5")
    write_hdf5(masks[0:20], images_path + "../border_masks_train.hdf5")
    write_hdf5(imgs[20:], images_path + "../imgs_test.hdf5")
    write_hdf5(ground_truth[20:], images_path + "../ground_truth_test.hdf5")
    write_hdf5(masks[20:], images_path + "../border_masks_test.hdf5")


def get_DRIVE():
    Nimgs = 20
    channels = 3
    height = 584
    width = 565

    def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, train_test="null"):
        imgs = np.empty((Nimgs, height, width, channels))
        groundTruth = np.empty((Nimgs, height, width))
        border_masks = np.empty((Nimgs, height, width))
        for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
            for i in range(len(files)):
                # original
                print("original image: " + files[i])
                img = Image.open(imgs_dir + files[i])
                imgs[i] = np.asarray(img)
                # corresponding ground truth
                groundTruth_name = files[i][0:2] + "_manual1.gif"
                print("ground truth name: " + groundTruth_name)
                g_truth = Image.open(groundTruth_dir + groundTruth_name)
                groundTruth[i] = np.asarray(g_truth)
                # corresponding border masks
                border_masks_name = ""
                if train_test == "train":
                    border_masks_name = files[i][0:2] + "_training_mask.gif"
                elif train_test == "test":
                    border_masks_name = files[i][0:2] + "_test_mask.gif"
                else:
                    print("specify if train or test!!")
                    exit()
                print("border masks name: " + border_masks_name)
                b_mask = Image.open(borderMasks_dir + border_masks_name)
                border_masks[i] = np.asarray(b_mask)

        print("imgs max: " + str(np.max(imgs)))
        print("imgs min: " + str(np.min(imgs)))
        assert (np.max(groundTruth) == 255 and np.max(border_masks) == 255)
        assert (np.min(groundTruth) == 0 and np.min(border_masks) == 0)
        print("ground truth and border masks are correctly withih pixel value range 0-255 (black-white)")
        # reshaping for my standard tensors
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        assert (imgs.shape == (Nimgs, channels, height, width))
        groundTruth = np.reshape(groundTruth, (Nimgs, 1, height, width))
        border_masks = np.reshape(border_masks, (Nimgs, 1, height, width))
        assert (groundTruth.shape == (Nimgs, 1, height, width))
        assert (border_masks.shape == (Nimgs, 1, height, width))
        return imgs, groundTruth, border_masks

    # ------------Path of the images --------------------------------------------------------------
    dataset_path = "./dataset/DRIVE/"
    # train
    original_imgs_train = "./dataset/DRIVE/training/images/"
    groundTruth_imgs_train = "./dataset/DRIVE/training/1st_manual/"
    borderMasks_imgs_train = "./dataset/DRIVE/training/mask/"
    # test
    original_imgs_test = "./dataset/DRIVE/test/images/"
    groundTruth_imgs_test = "./dataset/DRIVE/test/1st_manual/"
    borderMasks_imgs_test = "./dataset/DRIVE/test/mask/"
    # ---------------------------------------------------------------------------------------------

    # getting the training datasets
    imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train, groundTruth_imgs_train, borderMasks_imgs_train, "train")
    print("saving train datasets")
    write_hdf5(imgs_train, dataset_path + "imgs_train512.hdf5")
    write_hdf5(groundTruth_train, dataset_path + "ground_truth_train512.hdf5")
    write_hdf5(border_masks_train, dataset_path + "border_masks_train512.hdf5")

    # getting the testing datasets
    imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test, groundTruth_imgs_test, borderMasks_imgs_test, "test")
    print("saving test datasets")
    write_hdf5(imgs_test, dataset_path + "imgs_test512.hdf5")
    write_hdf5(groundTruth_test, dataset_path + "ground_truth_test512.hdf5")
    write_hdf5(border_masks_test, dataset_path + "border_masks_test512.hdf5")


def get_STARE():
    dataset_path = './datasets/STARE/'
    imgs_path = './datasets/STARE/images/'
    ground_truth_path = './datasets/STARE/labels_ah/'
    border_masks_path = './datasets/STARE/masks/'

    imgs = []
    ground_truth = []
    border_masks = []
    files = np.array(sorted(list(os.listdir(imgs_path))))
    for file in files:
        # corresponding ground truth
        ground_truth_name = file[:-4] + '.ah.ppm'
        print('ground truth name: ' + ground_truth_name)
        g_truth = np.array(image.load_img(ground_truth_path + ground_truth_name, grayscale=True))
        plt.imshow(g_truth, cmap='gray')
        plt.show()
        ground_truth.append(g_truth)
        # corresponding border masks
        border_masks_name = file
        print('border masks name: ' + border_masks_name)
        b_mask = np.array(image.load_img(border_masks_path + border_masks_name))
        plt.imshow(b_mask)
        plt.show()
        border_masks.append(b_mask[:, :, 0])
        # original
        print('original image: ' + file)
        img = np.asarray(image.load_img(imgs_path + file))
        plt.imshow(img)
        plt.show()
        b_mask = b_mask > 0
        plt.imshow(img * b_mask)
        plt.show()
        imgs.append(img * b_mask)

    print('imgs max: ' + str(np.max(imgs)) + ', min: ' + str(np.min(imgs)))
    assert (np.max(ground_truth) == 255 and np.max(border_masks) == 255)
    assert (np.min(ground_truth) == 0 and np.min(border_masks) == 0)
    print('ground truth and border masks are correctly withih pixel value range 0-255 (black-white)')
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    ground_truth = np.array(ground_truth)
    ground_truth = ground_truth.reshape(ground_truth.shape[0], 1, ground_truth.shape[1], ground_truth.shape[2])
    border_masks = np.array(border_masks)
    border_masks = border_masks.reshape(border_masks.shape[0], 1, border_masks.shape[1], border_masks.shape[2])

    print('imgs shape: ' + str(imgs.shape))
    print('ground_truth shape: ' + str(ground_truth.shape))
    print('border_masks shape: ' + str(border_masks.shape))

    print('saving train datasets')
    write_hdf5(imgs[0:10], dataset_path + 'imgs_train.hdf5')
    write_hdf5(ground_truth[0:10], dataset_path + 'ground_truth_train.hdf5')
    write_hdf5(border_masks[0:10], dataset_path + 'border_masks_train.hdf5')
    print('saving test datasets')
    write_hdf5(imgs[10:], dataset_path + 'imgs_test.hdf5')
    write_hdf5(ground_truth[10:], dataset_path + 'ground_truth_test.hdf5')
    write_hdf5(border_masks[10:], dataset_path + 'border_masks_test.hdf5')



def get_STARE2():
    dataset_path = './datasets/STARE/'
    imgs_path = './datasets/STARE/images/'
    ground_truth_path = './datasets/STARE/labels_ah/'
    border_masks_path = './datasets/STARE/masks/'

    imgs = []
    ground_truth = []
    border_masks = []
    files = np.array(sorted(list(os.listdir(imgs_path))))
    for file in files:
        # corresponding ground truth
        ground_truth_name = file[:-4] + '.ah.ppm'
        print('ground truth name: ' + ground_truth_name)
        g_truth = np.array(image.load_img(ground_truth_path + ground_truth_name, grayscale=True))
        #plt.imshow(g_truth, cmap='gray')
        #plt.show()
        ground_truth.append(g_truth)
        # corresponding border masks
        border_masks_name = file
        print('border masks name: ' + border_masks_name)
        b_mask = np.array(image.load_img(border_masks_path + border_masks_name))
        #plt.imshow(b_mask)
        #plt.show()
        border_masks.append(b_mask[:, :, 0])
        # original
        print('original image: ' + file)
        img = np.asarray(image.load_img(imgs_path + file))
        #plt.imshow(img)
        #plt.show()
        b_mask = b_mask > 0
        #plt.imshow(img * b_mask)
        #plt.show()
        imgs.append(img * b_mask)

    print('imgs max: ' + str(np.max(imgs)) + ', min: ' + str(np.min(imgs)))
    assert (np.max(ground_truth) == 255 and np.max(border_masks) == 255)
    assert (np.min(ground_truth) == 0 and np.min(border_masks) == 0)
    print('ground truth and border masks are correctly withih pixel value range 0-255 (black-white)')
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    ground_truth = np.array(ground_truth)
    ground_truth = ground_truth.reshape(ground_truth.shape[0], 1, ground_truth.shape[1], ground_truth.shape[2])
    border_masks = np.array(border_masks)
    border_masks = border_masks.reshape(border_masks.shape[0], 1, border_masks.shape[1], border_masks.shape[2])

    print('imgs shape: ' + str(imgs.shape))
    print('ground_truth shape: ' + str(ground_truth.shape))
    print('border_masks shape: ' + str(border_masks.shape))

    print('saving datasets')
    print(imgs.shape)
    print(ground_truth.shape)
    print(border_masks.shape)
    write_hdf5(imgs, dataset_path + 'imgs.hdf5')
    write_hdf5(ground_truth, dataset_path + 'ground_truth.hdf5')
    write_hdf5(border_masks, dataset_path + 'border_masks.hdf5')


if __name__ == '__main__':
    # get_HRF()
    # get_STARE2()
    # get_STARE()
    dataset_path = './datasets/STARE/'
    imgs_path = './datasets/STARE/images/'
    files = np.array(sorted(list(os.listdir(imgs_path))))
    print(files)