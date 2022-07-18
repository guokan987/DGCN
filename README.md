Updata 1:

2022/7/18
Please Emails:guokan.cn@gmail.com to contact me;
邮箱地址以更换为：guokan.cn@gmail.com。

The public code of DGCN in T-ITS with Pytorch 1.2.0 based on Titan RTX 24G GPU

This is a document for this code++++++++++++++++++++++++++++++++++++++++++++++++++++++++=>

***First, the structure of the code:

.lib|->data_preparation.py: read data and create three part: training-set, validation-set and test-set : 60%/20%/20%

.lib|->utils.py: some functions for data_preparation.py and metrics: validation_loss and test-set evalution between prediction and ground truth.

.lib|-> metrics.py: serve for utils.py

.utils.py: we construct all models' base function block in this file, and for example, DGCN's block is ST_BLOCK_2;

.models.py: we construct all models in this file based on blocks in the utils.py.

.DGCN.py /ASTGCN.py /DGCN_R.py /DGCN_Mask.py /DGCN_Res.py /DGCN_GAT.py : these files are main file when program run, in these files, we mainly write main function of neural network in Pytorch.

. _prediction_04/08.npz: these npz files save several output parts of NN: prediction, test-set's ground truth , road network's graph laplace matrices.

***How to run these files?

in Jupyter Notebook:

you can run two public datasets: PeMSD4 and PeMSD8:

(1) PeMSD4: %run DGCN.py/... (or other main files) --data_name 4 --num_point 307

(2) PeMSD8: %run DGCN.py/... (or other main files) --data_name 8 --num_point 170

***if you have any questions:

Please leave a message in the issue area with English or Chinese. You can also email me!

***The each epoch's save Pytorch model (stat_dict) file in the following website, if you are interested in each epoch's details in our model training phase, please see it!

weibsite：https://pan.baidu.com/s/1EVYLbO3V0ulFsDjpcMMM2g key：kzey

Our new work has accepted by AAAI2021, if you have interest, please go to https://github.com/guokan987/HGCN.
