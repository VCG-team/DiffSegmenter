
<!-- This is code for paper "Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter"

# Open Vocabulary Semantic Segmentation
The open_vocabulary folder contains code for open vocabulary semantic segmentation. It includes scripts for the voc, coco, and Pascal context datasets. Running scripts in this folder will generate segmentation results for the respective datasets.
To obtain the final mean intersection over union (miou), run the evaluation script on the segmentation results.

Taking the voc10 dataset as an example:
1. .json files contain open vocabulary labels predicted based on blip and clip. If it is weakly supervised semantic segmentation, these predicted labels are not required.
2. ptp_stable_voc10.py is used to predict semantic segmentation results based on the labels.
3. evaluation_voc10.py is used to evaluate the semantic segmentation results. -->
# Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter

By [Jinglong Wang](https://github.com/wangjl-nb), [Xiawei Li](https://github.com/Sunny599), [Jing zhang](https://scholar.google.com.hk/citations?user=XtwOoQgAAAAJ&hl=zh-CN&oi=ao), [Qiangyuan Xu](https://github.com/xu7yue), [Qin Zhou](https://github.com/Matrix53), [Qian Yu](https://scholar.google.com.hk/citations?user=mmm90qgAAAAJ&hl=zh-CN&oi=ao), [Sheng Lu](https://scholar.google.com.hk/citations?user=_8lB7xcAAAAJ&hl=zh-CN&oi=ao), [Dong Xu](https://scholar.google.com.hk/citations?user=7Hdu5k4AAAAJ&hl=zh-CN&oi=ao).

This repository is an official implementation of the paper [Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter](https://arxiv.org/abs/2309.02773). And you're welcome to our [project page](https://vcg-team.github.io/DiffSegmenter-webpage/).

## New Paper🎉
We are thrilled to announce our latest paper "ELBO-T2IAlign: A Generic ELBO-Based Method for Calibrating Pixel-level Text-Image Alignment in Diffusion Models", which explores the impact of original diffusion loss function. This work builds upon this repo and offers new insights into downstream tasks of diffusion models. Check it out [here](https://github.com/VCG-team/elbo-t2ialign) and explore how it enhances our understanding of diffusion model.

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



## Installation

### Requirements

* Linux, CUDA>=11.7, GCC>=9.4
  
* Python>=3.8

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n ldm python=3.8
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

Please download datasets and organize them as following:

```

├── COCO2014
│   ├── annotations
│   ├── coco_seg_anno
│   ├── images
│   │   ├── test2014
│   │   ├── train2014
│   │   └── val2014
│   └── mask
│       ├── train2014
│       └── val2014


└── VOCdevkit
    ├── VOC2010
    │   ├── Annotations
    │   ├── ImageSets
    │   │   ├── Action
    │   │   ├── Layout
    │   │   ├── Main
    │   │   ├── Segmentation
    │   │   └── SegmentationContext
    │   ├── JPEGImages
    │   ├── SegmentationClass
    │   ├── SegmentationClassContext
    │   └── SegmentationObject
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        │   ├── Action
        │   ├── Layout
        │   ├── Main
        │   └── Segmentation
        ├── JPEGImages
        ├── SegmentationClass
        ├── SegmentationClassAug
        └── SegmentationObject
```

### Open Vocabulary Semantic Segmentation

#### Evaluation

For the setting of Open Vocabulary Semantic Segmentation， our model does not require training; it directly produces segmentation results.


The ‘open_vocabulary’ folder contains code for open vocabulary semantic segmentation. It includes scripts for the voc, coco, and Pascal context datasets.

Taking the voc10 dataset as an example:

Step 1: Modify your dataset path in the Python file.

Step 2: Run ptp_stable_voc10.py to generate segmentation results.

```
python ptp_stable_voc10.py
```

Step 3: Run the evaluation script, remember to update the file path. MIoU will be recorded in eval.txt

```
python evaluation_voc10.py
```
