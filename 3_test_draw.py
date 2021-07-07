import argparse
import torch.nn.functional as F
#from matplotlib import pyplot as plt
from utils.preprocessing import *
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score

import torch.backends.cudnn as cudnn

from utils.misc import *
import models

model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='CNN')
# use last save model
parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
parser.add_argument('--stride_size', type=int, default=5, help='stride size (default: 5)')
parser.add_argument('--batch_size', type=int, default=32, help='stride size (default: 1024)')
parser.add_argument('--threshold_confusion', default=0.5, type=float, help='threshold_confusion')
parser.add_argument('--check_path', type=str, help='load model path')

args = vars(parser.parse_args())

checkpoint = torch.load(args['check_path'])
checkpoint['args']['device'] = args['device']
checkpoint['args']['check_path'] = args['check_path']
checkpoint['args']['stride_size'] = args['stride_size']
checkpoint['args']['threshold_confusion'] = args['threshold_confusion']
checkpoint['args']['batch_size'] = args['batch_size']
checkpoint['args']['data_path'] = './datasets/'
args = checkpoint['args']
max_acc = checkpoint['max_acc']
max_sensitivity = checkpoint['max_sensitivity']
max_F1_score = checkpoint['max_F1_score']
cur_epoch = checkpoint['epoch'] + 1
logs = checkpoint['logs']
threshold_confusion = args['threshold_confusion']
print(logs[-1])
print(args)

cudnn.benchmark = True
torch.cuda.is_available()
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
print("os.environ['CUDA_VISIBLE_DEVICES']: ", os.environ['CUDA_VISIBLE_DEVICES'])

if args['model'] == 'N2UNet_v5':
    args['model'] = 'M3FCN'
net = models.__dict__[args['model']]()
net.eval().cuda()
net.load_state_dict(checkpoint['net'], strict=False)

data_path = args['data_path'] + args['dataset'] + '/'
imgs_original = load_hdf5(data_path + 'imgs.hdf5')
#imgs = preprocessing(imgs_original)
imgs = rgb2gray(imgs_original)
if args['patch_size'] == '48':
    imgs = rgb2gray(imgs_original)
else:
    imgs = get_green(imgs_original)
imgs = dataset_normalized(imgs)
imgs = clahe_equalized(imgs)
imgs = adjust_gamma(imgs, 1.2)
imgs = imgs / 255.

gts = load_hdf5(data_path + 'ground_truth.hdf5')
gts = gts / 255.
masks = load_hdf5(data_path + 'border_masks.hdf5')

test_imgs = imgs[args['fold']].reshape((1, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
test_gts = gts[args['fold']].reshape((1, gts.shape[1], gts.shape[2], gts.shape[3]))
test_masks = masks[args['fold']].reshape((1, masks.shape[1], masks.shape[2], masks.shape[3]))
print(test_imgs.shape)
patch_size = args['patch_size']
stride_size = args['stride_size']
h = 0
mkdir_p('pred/')
for t in range(len(test_imgs)):
    test_img = test_imgs[t].reshape((1, test_imgs.shape[1], test_imgs.shape[2], test_imgs.shape[3]))
    test_img = paint_border_overlap(test_img, patch_size, stride_size)
    print(test_img.shape)
    img_h = test_img.shape[2]  # height of the full image
    img_w = test_img.shape[3]  # width of the full image
    
    preds = []
    patches = []
    for i in range(test_img.shape[0]):  # loop over the full images
        for h in range((img_h - patch_size) // stride_size + 1):
            for w in range((img_w - patch_size) // stride_size + 1):
                patch = test_img[i, :, h * stride_size:(h * stride_size) + patch_size, w * stride_size:(w * stride_size) + patch_size]
                patches.append(patch)
                if len(patches) == args['batch_size']:
                    print(np.asarray(patches).shape)
                    test_set = TestDataset(np.asarray(patches))
                    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=0)

                    for batch_idx, inputs in enumerate(test_loader):
                        inputs = inputs.cuda()
                        print('inputs shape, ', inputs.shape)
                        outputs, outs = net(inputs)
                        np.save('pred/' + str(h) + '.npy', outs)
                        outs = outs.reshape((outs.shape[0] * outs.shape[1], outs.shape[1]))
                        if np.max(outs) > 1:
                            img = Image.fromarray(outs.astype(np.uint8))  # the image is already 0-255
                        else:
                            img = Image.fromarray((outs * 255).astype(np.uint8))  # the image is between 0-1
                        img.save('pred/' + str(h) + '.png')
                        h+=1
                
                    patches = []