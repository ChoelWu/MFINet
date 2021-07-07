import numpy as np
import os
import torch
import matplotlib.pyplot as plt

print('==========================================预测图像拼接开始==========================================')

# /home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/STARE/UNet/1/checkpoints/last_result_average
# /home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/STARE/UNetFG/20/checkpoints/periods/7_result_average/pred_imgs.npy

log_path = '/home/sdc_3_7T/jiangyun/wuchao/MSINet/logs/STARE/UNetPlusPlus/'

pred_imgs = ''

for i in range(20):
    single_check_path = log_path + str(i + 1) + '/checkpoints/periods/'
    f_list = os.listdir(single_check_path)
    for dbtype in f_list:
        if os.path.isdir(os.path.join(single_check_path, dbtype)):
            single_pre_img_path = os.path.join(single_check_path, dbtype) + '/pred_imgs.npy'
            pred_img = np.load(single_pre_img_path)
            if 0 == i:
                pred_imgs = pred_img
            else:
                pred_imgs = np.concatenate((pred_imgs, pred_img), 0)
            break

np.save(log_path + 'pred_imgs.npy', pred_imgs)

print('==========================================预测图像拼接完成==========================================')
print('生成文件路径：' + os.path.abspath(log_path + 'pred_imgs.npy'))
