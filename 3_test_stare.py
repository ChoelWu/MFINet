import argparse
import datetime
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
parser.add_argument('--batch_size', type=int, default=128, help='stride size (default: 1024)')
parser.add_argument('--threshold_confusion', default=0.5, type=float, help='threshold_confusion')
parser.add_argument('--check_path', type=str, help='load model path')

args = vars(parser.parse_args())

checkpoint = torch.load(args['check_path'])
checkpoint['args']['device'] = args['device']
checkpoint['args']['check_path'] = args['check_path']
checkpoint['args']['stride_size'] = args['stride_size']
checkpoint['args']['threshold_confusion'] = args['threshold_confusion']
checkpoint['args']['batch_size'] = args['batch_size']
checkpoint['args']['data_path'] = '/home/sdc_3_7T/jiangyun/wuchao/dataset/retinal_vessels/'
args = checkpoint['args']
max_acc = checkpoint['max_acc']
max_sensitivity = checkpoint['max_sensitivity']
max_F1_score = checkpoint['max_F1_score']
cur_epoch = checkpoint['epoch'] + 1
logs = checkpoint['logs']
threshold_confusion = args['threshold_confusion']
print(logs[-1])
print(args)

path_arr = args['check_path'].split('/')
basic_path = args['check_path'][:-len(path_arr[-1])] + path_arr[-1].split('.')[0]
basic_path = basic_path + '_result_average/'
mkdir_p(basic_path + 'img_mask_pred/')

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

patch_size = args['patch_size']
stride_size = args['stride_size']
if os.path.exists(basic_path + 'pred_imgs.npy') is False:
    pred_imgs = []
    for t in range(len(test_imgs)):
        test_img = test_imgs[t].reshape((1, test_imgs.shape[1], test_imgs.shape[2], test_imgs.shape[3]))
        test_img = paint_border_overlap(test_img, patch_size, stride_size)
        img_h = test_img.shape[2]  # height of the full image
        img_w = test_img.shape[3]  # width of the full image
        
        preds = []
        patches = []
        # 分割开始时间
        starttime = datetime.datetime.now()
        for i in range(test_img.shape[0]):  # loop over the full images
            for h in range((img_h - patch_size) // stride_size + 1):
                for w in range((img_w - patch_size) // stride_size + 1):
                    patch = test_img[i, :, h * stride_size:(h * stride_size) + patch_size, w * stride_size:(w * stride_size) + patch_size]
                    patches.append(patch)
                    if len(patches) == args['batch_size']:
                        test_set = TestDataset(np.asarray(patches))
                        test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=0)

                        for batch_idx, inputs in enumerate(test_loader):
                            inputs = inputs.cuda()
                            outputs = net(inputs)
                            outputs = torch.nn.functional.softmax(outputs, dim=1)
                            outputs = outputs.permute(0, 2, 3, 1)
                            outputs = outputs.view(-1, outputs.shape[1] * outputs.shape[2], 2)
                            outputs = outputs.data.cpu().numpy()
                            preds.append(outputs)

                        patches = []
        
        if len(patches) > 0:
            test_set = TestDataset(np.asarray(patches))
            test_loader = DataLoader(test_set, batch_size=len(patches), shuffle=False, num_workers=0)

            for batch_idx, inputs in enumerate(test_loader):
                inputs = inputs.cuda()
                outputs = net(inputs)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                outputs = outputs.permute(0, 2, 3, 1)
                outputs = outputs.view(-1, outputs.shape[1] * outputs.shape[2], 2)
                outputs = outputs.data.cpu().numpy()
                preds.append(outputs)
        
        print(np.asarray(preds).shape)
        preds = np.concatenate(preds, axis=0)
        print(preds.shape)
        pred_patches = pred_to_imgs(preds, args['patch_size'])
        pred_img = recompone_overlap(pred_patches, img_h, img_w, stride_size)
        pred_imgs.append(pred_img[:, :, 0:test_imgs.shape[2], 0:test_imgs.shape[3]][0])

        # 分割结束时间
        endtime = datetime.datetime.now()

        print('----------time marker---------------')
        print (endtime - starttime)
        print('----------time marker---------------')

    pred_imgs = np.array(pred_imgs)
    np.save(basic_path + 'pred_imgs.npy', pred_imgs)
else:
    pred_imgs = np.load(basic_path + 'pred_imgs.npy')

kill_border(pred_imgs, test_masks)  # DRIVE MASK  #only for visualization
# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
print("test imgs shape: " + str(test_imgs.shape))
print("pred imgs shape: " + str(pred_imgs.shape))
print("test masks shape: " + str(test_gts.shape))
N_visual = 1
# visualize(group_images(test_imgs, N_visual), basic_path + 'all_originals.eps')
visualize(group_images(pred_imgs, N_visual), basic_path + 'all_predictions.png')
# visualize(group_images(test_gts, N_visual), basic_path + 'all_groundTruths.png')
# visualize results comparing mask and prediction:
assert (test_imgs.shape[0] == pred_imgs.shape[0] and test_imgs.shape[0] == test_gts.shape[0])
N_predicted = test_imgs.shape[0]
group = N_visual
assert (N_predicted % group == 0)
for i in range(int(N_predicted / group)):
    orig_stripe = group_images(test_imgs[i * group:(i * group) + group, :, :, :], group)
    masks_stripe = group_images(test_gts[i * group:(i * group) + group, :, :, :], group)
    pred_stripe = group_images(pred_imgs[i * group:(i * group) + group, :, :, :], group)
    total_img = np.concatenate((orig_stripe, masks_stripe, pred_stripe), axis=0)
    visualize(total_img, basic_path + "img_mask_pred/" + str(i) + '.eps')  # .show()

# ====== Evaluate the results
print("\n\n========  Evaluate the results =======================")
# predictions only inside the FOV
y_scores = np.asarray(pred_imgs).astype(np.float32).flatten()
y_true = np.asarray(test_gts).astype(np.float32).flatten()
#y_scores, y_true = pred_only_FOV(pred_imgs, test_gts, test_masks)  # returns data only inside the FOV

# Area under the ROC curve
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
# test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
print("\nArea under the ROC curve: " + str(AUC_ROC))
"""
roc_curve = plt.figure()
plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
plt.title('ROC curve')
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (True Positive Rate)")
plt.legend(loc="lower right")
plt.savefig(basic_path + "ROC.eps")
"""
# Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision, recall)
print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
"""
prec_rec_curve = plt.figure()
plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(basic_path + "Precision_recall.eps")
"""
# Confusion matrix
#threshold_confusion = threshold_confusion
print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
y_pred = np.empty((y_scores.shape[0]))
for i in range(y_scores.shape[0]):
    y_pred[i] = 1 if y_scores[i] >= threshold_confusion else 0

confusion = confusion_matrix(y_true, y_pred)
print(confusion)
accuracy = 0
if float(np.sum(confusion)) != 0:
    accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
print("Global Accuracy: " + str(accuracy))
specificity = 0
if float(confusion[0, 0] + confusion[0, 1]) != 0:
    specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
print("Specificity: " + str(specificity))
sensitivity = 0
if float(confusion[1, 1] + confusion[1, 0]) != 0:
    sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
print("Sensitivity: " + str(sensitivity))

# F1 score
F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
print("\nF1 score (F-measure): " + str(F1_score))

# Save the results
file_perf = open(basic_path + 'performances.txt', 'w')
file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                + "\nF1 score (F-measure): " + str(F1_score)
                + "\nACCURACY: " + str(accuracy)
                + "\nSENSITIVITY: " + str(sensitivity)
                + "\nSPECIFICITY: " + str(specificity)
                + "\n\nConfusion matrix:"
                + str(confusion)
                )
file_perf.close()
