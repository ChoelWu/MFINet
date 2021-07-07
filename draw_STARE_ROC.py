import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
from utils.misc import *

data_path = './datasets/STARE/'  # 数据集路径

font = {'size': 14}
font_title = {'size': 16}
font_legend = {'size': 12}

test_gts = load_hdf5(data_path + 'ground_truth.hdf5')
test_gts = test_gts / 255.
test_masks = load_hdf5(data_path + 'border_masks.hdf5')

model_path_UNet = '/home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/STARE/UNet/'
pred_imgs_UNet = np.load(model_path_UNet + 'pred_imgs.npy')

model_path_UNetMI = '/home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/STARE/UNetMI/'
pred_imgs_UNetMI = np.load(model_path_UNet + 'pred_imgs.npy')

model_path_UNetFG = '/home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/STARE/UNetFG/'
pred_imgs_UNetFG = np.load(model_path_UNetFG + 'pred_imgs.npy')

model_path_UNetMR = '/home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/STARE/UNetMR/'
pred_imgs_UNetMR = np.load(model_path_UNetMR + 'pred_imgs.npy')

model_path_UNetMRFG = '/home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/STARE/UNetMRFG/'
pred_imgs_UNetMRFG = np.load(model_path_UNetMRFG + 'pred_imgs.npy')

# AUC_ROC
plt.figure(1)

y_scores_UNet = np.asarray(pred_imgs_UNet).astype(np.float32).flatten()
y_true_UNet = np.asarray(test_gts).astype(np.float32).flatten()
fpr_UNet, tpr_UNet, thresholds_UNet = roc_curve((y_true_UNet), y_scores_UNet)
AUC_ROC_UNet = roc_auc_score(y_true_UNet, y_scores_UNet)

y_scores_UNetMI = np.asarray(pred_imgs_UNetMI).astype(np.float32).flatten()
y_true_UNetMI = np.asarray(test_gts).astype(np.float32).flatten()
fpr_UNetMI, tpr_UNetMI, thresholds_UNetMI = roc_curve((y_true_UNetMI), y_scores_UNetMI)
AUC_ROC_UNetMI = roc_auc_score(y_true_UNetMI, y_scores_UNetMI)

y_scores_UNetFG = np.asarray(pred_imgs_UNetFG).astype(np.float32).flatten()
y_true_UNetFG = np.asarray(test_gts).astype(np.float32).flatten()
fpr_UNetFG, tpr_UNetFG, thresholds_UNetFG = roc_curve((y_true_UNetFG), y_scores_UNetFG)
AUC_ROC_UNetFG = roc_auc_score(y_true_UNetFG, y_scores_UNetFG)

y_scores_UNetMR = np.asarray(pred_imgs_UNetMR).astype(np.float32).flatten()
y_true_UNetMR = np.asarray(test_gts).astype(np.float32).flatten()
fpr_UNetMR, tpr_UNetMR, thresholds_UNetMR = roc_curve((y_true_UNetMR), y_scores_UNetMR)
AUC_ROC_UNetMR = roc_auc_score(y_true_UNetMR, y_scores_UNetMR)

y_scores_UNetMRFG = np.asarray(pred_imgs_UNetMRFG).astype(np.float32).flatten()
y_true_UNetMRFG = np.asarray(test_gts).astype(np.float32).flatten()
fpr_UNetMRFG, tpr_UNetMRFG, thresholds_UNetMRFG = roc_curve((y_true_UNetMRFG), y_scores_UNetMRFG)
AUC_ROC_UNetMRFG = roc_auc_score(y_true_UNetMRFG, y_scores_UNetMRFG)

plt.plot(fpr_UNet, tpr_UNet, label='UNet (AUC = %0.4f)' % AUC_ROC_UNet, color='black')
plt.plot(fpr_UNet, tpr_UNetMI, label='UNet+MI (AUC = %0.4f)' % AUC_ROC_UNetMI, color='yellow')
plt.plot(fpr_UNetFG, tpr_UNetFG, label='UNet+FG (AUC = %0.4f)' % AUC_ROC_UNetFG, color='red')
plt.plot(fpr_UNetMR, tpr_UNetMR, label='UNet+MR (AUC = %0.4f)' % AUC_ROC_UNetMR, color='green')
plt.plot(fpr_UNetMRFG, tpr_UNetMRFG, label='UNet+FG+MR (AUC = %0.4f)' % AUC_ROC_UNetMRFG, color='blue')

plt.xlabel("FPR (False Positive Rate)", font)
plt.ylabel("TPR (True Positive Rate)", font)
plt.legend(loc="lower right", prop=font_legend)
plt.axis([0, 1, 0.94, 1])
plt.title('STARE ROC curve', font_title)
plt.savefig("./pred/STARE_AUC_ROC.eps")
plt.show()

# Precision Recall (PR) curve
plt.figure(2)

precision_UNet, recall_UNet, thresholds_UNet = precision_recall_curve(y_true_UNet, y_scores_UNet)
AUC_prec_rec_UNet = auc(precision_UNet, recall_UNet, reorder=True)

precision_UNetMI, recall_UNetMI, thresholds_UNetMI = precision_recall_curve(y_true_UNetMI, y_scores_UNetMI)
AUC_prec_rec_UNetMI = auc(precision_UNetMI, recall_UNetMI, reorder=True)

precision_UNetFG, recall_UNetFG, thresholds_UNetFG = precision_recall_curve(y_true_UNetFG, y_scores_UNetFG)
AUC_prec_rec_UNetFG = auc(precision_UNetFG, recall_UNetFG, reorder=True)

precision_UNetMR, recall_UNetMR, thresholds_UNetMR = precision_recall_curve(y_true_UNetMR, y_scores_UNetMR)
AUC_prec_rec_UNetMR = auc(precision_UNetMR, recall_UNetMR, reorder=True)

precision_UNetMRFG, recall_UNetMRFG, thresholds_UNetMRFG = precision_recall_curve(y_true_UNetMRFG, y_scores_UNetMRFG)
AUC_prec_rec_UNetMRFG = auc(precision_UNetMRFG, recall_UNetMRFG, reorder=True)

plt.plot(recall_UNet, precision_UNet, '-', label='UNet (AUC = %0.4f)' % AUC_prec_rec_UNet, color='black')
plt.plot(recall_UNetMI, precision_UNetMI, '-', label='UNet+MI (AUC = %0.4f)' % AUC_prec_rec_UNetMI, color='yellow')
plt.plot(recall_UNetFG, precision_UNetFG, '-', label='UNet+FG (AUC = %0.4f)' % AUC_prec_rec_UNetFG, color='red')
plt.plot(recall_UNetMR, precision_UNetMR, '-', label='UNet+MR (AUC = %0.4f)' % AUC_prec_rec_UNetMR, color='green')
plt.plot(recall_UNetMRFG, precision_UNetMRFG, '-', label='UNet+FG+MR (AUC = %0.4f)' % AUC_prec_rec_UNetMRFG, color='blue')

plt.title('STARE Precision Recall curve', font_title)
plt.xlabel("Recall", font)
plt.ylabel("Precision", font)
plt.legend(loc="lower left", prop=font_legend)
plt.axis([0, 1, 0.6, 1])
plt.savefig("./pred/STARE_AUC_PR.eps")
plt.show()
