# MFI-Net: A multi-resolution fusion input network for retinal vessel segmentation

> **Authors:** 
> Yun Jiang,
> Chao Wu,
> Ge Wang,
> Hui-Xia Yao,
> Wen-Huan Liu

## 0. Preface

- This repository provides code for "_**MFI-Net: A multi-resolution fusion input network for retinal vessel segmentation**_" PLOS ONE. 
(https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0253056) 

## 1. MFI-Net

<p align="center">
    <img src="https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0253056.g001"/> <br />
    <em> 
    Figure 1. Overview of MFI-Net segmentation model for retinal vessel. 
    </em>
</p>

## 2. Dataset

<p align="center">
    <img src="https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0253056.g003"/> <br />
    <em> 
    Figure 2. Sample images of DRIVE, CHASE DB1 and STARE datasets. 
    </em>
</p>


- **DRIVE**
    DRIVE dataset files are available from http://www.isi.uu.nl/Research/Databases/DRIVE.
- **CHASE_DB1**
    CHASE_DB1 dataset files are available from https://blogs.kingston.ac.uk/retinal/chasedb1/
- **CHASE_DB1**
     STARE dataset files are available from http://cecas.clemson.edu/~ahoover/stare/.

## 3. Usage
1. Train

`python 2_train_dynamic.py --device 0 --dataset DRIVE --data_path /home/data/dataset/ --model UNet --epoch 200 --batch_size 1024 --patch_num 10000 --logs_path /home/data/logs/ --lr 0.001`

2. Test

`python 3_test.py --device 0 --check_path YOUR_LOG_PATH`

## 4. Result

<p align="center">
    <img src="https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0253056.t001"/> <br />
    <em> 
    Table 1. Ablation experiment results. 
    </em>
</p>

<p align="center">
    <img src="https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0253056.g006"/> <br />
    <em> 
    Figure 3. Segmentation result of ablation experiment. 
    </em>
</p>



<p align="center">
    <img src="https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0253056.g007"/> <br />
    <em> 
    Figure 4. Comparison of segmentation results of MFI-Net(ours), UNet++ and AA-UNet. 
    </em>
</p>

## 5. Our related work
A Multi-Scale Residual Attention Network for Retinal Vessel Segmentation. (https://www.mdpi.com/2073-8994/13/1/24)

Multi-Scale and Multi-Branch Convolutional Neural Network for Retinal Image Segmentation. (https://www.mdpi.com/2073-8994/13/3/365)

**[⬆ back to top](#0-preface)**
