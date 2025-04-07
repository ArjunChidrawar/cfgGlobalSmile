# INCLG: A Multi-task Image Inpainting Approach to Non-cleft Lip Generation

![Image text](https://github.com/ChrisChen1023/INCLG/blob/main/software_overview6.png)

=================================================================================

Patients with cleft lips usually need to undergo surgical treatment by a professional surgeon to improve nasolabial appearance. We propose a novel multi-task architecture to implement image inpainting and facial landmark prediction with interactive parameter sharing, the architecture is designed with Pytorch. Our system is trained with irregular masks so that the mask can fit different cleft lip types in testing. As cleft lip data is sensitive and not ready to be released, we only use cleft lip images for testing to protect patients privacy. 

Due to the sensitivity of patient privacy, we are not allowed to use patients images on Code Ocean. In this case, we only reproduce our results on non-cleft lips human face (CelebA).

Our code is reproducible directely on: [Code Ocean](https://codeocean.com/capsule/1102688/tree/v1)

**Initialization**
--------------------
Python>=3.7

PyTorch

requirements.txt

**Dataset**
--------------------
For the full CelebA dataset, please refer to http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

For the irrgular mask dataset, please refer to http://masc.cs.gmu.edu/wiki/partialconv

For the landmarks, please use https://github.com/1adrianb/face-alignment to generate landamrks as ground truth.

Please use `script/flist.py` to create `.flist` file for training and testing.



**Pre-trained model**
--------------------
We released the pre-trained model 
CelebA: [Google Drive](https://drive.google.com/drive/folders/1H9FZ-jJUkYBDcNASX8kBnmipgGgv_y7t?usp=sharing)

**Getting Started**
----------------------
Download the pre-trained model to `./checkpoints`

Set your own `config.yml` file and copy it to corresponding checkpoint folder, run:
```
python train.py
```
For testing, run:
```
python test.py
```


