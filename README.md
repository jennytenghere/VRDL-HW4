# VRDL-HW4
StudentID:
Name:
## Introduction
This is the fourth assignment of the VRDL course. The task is to apply PromptIR on a dataset contain rain and snow degradation. The goal is to remove this degradation so that the PSNR of the recovered images is as high as possible. I found the original Prompt Generation Block is lacking interpretability. The dimension of the Prompt Components are N*H*W*C, meaning that different positions in the image correspond to different cell of prompt components. However, noise and degradation should not vary simply because they appear in different locations of the image. Therefore, I tried several approaches: for example, making he prompt more global to improve interpretability, or making the prompt more localized by generating different prompts based on the features of each patch.  
## How to install and use
### Install environment
conda env create -f vrdl.yaml
### Directory structure (***Please add or replace the files following the directory structure used in the PromptIR GitHub repository***)

```
.  
├── PromptIR/  
│   ├── train_ckpt  
│   └── train_v8.py  
├── hw4_realse_dataset/  
│   ├── test/  
│   └── train/  
└── example_img2npz.py  
```
### Run training
python train.py
### Run test
python test.py
### Create submission file
python example_img2npz.py


## Performance
![image](https://github.com/user-attachments/assets/996291d6-154f-482d-b22a-6c0666d69903)
