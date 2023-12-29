
<!-- This is code for paper "Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter"

# Open Vocabulary Semantic Segmentation
The open_vocabulary folder contains code for open vocabulary semantic segmentation. It includes scripts for the voc, coco, and Pascal context datasets. Running scripts in this folder will generate segmentation results for the respective datasets.
To obtain the final mean intersection over union (miou), run the evaluation script on the segmentation results.

Taking the voc10 dataset as an example:
1. .json files contain open vocabulary labels predicted based on blip and clip. If it is weakly supervised semantic segmentation, these predicted labels are not required.
2. ptp_stable_voc10.py is used to predict semantic segmentation results based on the labels.
3. evaluation_voc10.py is used to evaluate the semantic segmentation results. -->
# Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter

By [Jinglong Wang],  [Xiawei Li],  [Jing zhang](https://scholar.google.com.hk/citations?user=XtwOoQgAAAAJ&hl=zh-CN&oi=ao), [Qingyuan Xu], [YuQian], [Sheng Lu],[Dong Xu].

This repository is an official implementation of the paper [Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter](https://arxiv.org/abs/2309.02773).


## Citing DiffSegmenter
If you find DiffSegmenter useful in your research, please consider citing:
```bibtex
@misc{wang2023diffusion,
      title={Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter}, 
      author={Jinglong Wang and Xiawei Li and Jing Zhang and Qingyuan Xu and Qin Zhou and Qian Yu and Lu Sheng and Dong Xu},
      year={2023},
      eprint={2309.02773},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!-- 
*Note:*

1. All models of Deformable DETR are trained with total batch size of 32. 
2. Training and inference speed are measured on NVIDIA Tesla V100 GPU.
3. "Deformable DETR (single scale)" means only using res5 feature map (of stride 32) as input feature maps for Deformable Transformer Encoder.
4. "DC5" means removing the stride in C5 stage of ResNet and add a dilation of 2 instead.
5. "DETR-DC5+" indicates DETR-DC5 with some modifications, including using Focal Loss for bounding box classification and increasing number of object queries to 300.
6. "Batch Infer Speed" refer to inference with batch size = 4  to maximize GPU utilization.
7. The original implementation is based on our internal codebase. There are slight differences in the final accuracy and running time due to the plenty details in platform switch. -->


## Installation

### Requirements

* Linux, CUDA>=11.5, GCC>=9.4
  
* Python>=3.8

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n ldm python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate ldm
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```


## Usage

### Dataset preparation

Please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### Open Vocabulary Semantic Segmentation

#### Evaluation

For the setting of Open Vocabulary Semantic Segmentation， our model does not require training; it directly produces segmentation results.
The ‘open_vocabulary’ folder contains code for open vocabulary semantic segmentation. It includes scripts for the voc, coco, and Pascal context datasets. Running scripts in this folder will generate segmentation results for the respective datasets.
To obtain the final mean intersection over union (miou), run the evaluation script on the segmentation results.

Taking the voc10 dataset as an example:
step 1: Modify your dataset path in the Python file.
Step 2: Run ptp_stable_voc10.py to generate segmentation results.
'''
python ptp_stable_voc10.py
'''
Step 3: Run the evaluation script, remember to update the file path. MIoU will be recorded in eval.txt
'''
python evaluation_cvoc10.py
'''
