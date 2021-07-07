from sklearn.metrics import f1_score

from utils.misc import *

data_path = '/home/jiangyun/wuchao/MSINet/datasets/DRIVE/'

test_gts = load_hdf5(data_path + 'ground_truth_test.hdf5')
test_gts = test_gts / 255.

pred_imgs = np.load('/home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/DRIVE/UNetMI/2/checkpoints/periods/30_result_average/pred_imgs.npy')

print('===========================================================')

for i in range(pred_imgs.shape[0]):
    y_scores = np.asarray(pred_imgs[i]).astype(np.float32).flatten()
    y_true = np.asarray(test_gts[i]).astype(np.float32).flatten()

    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        y_pred[i] = 1 if y_scores[i] >= 0.5 else 0

    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)

    print(F1_score)

print('===========================================================')
