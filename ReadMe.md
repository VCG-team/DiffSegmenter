<!-- # 可视化
visual_code是可视化的代码
# 开放词表语义分割
open_vocabulary文件夹是开放词表分割的代码。包括voc、coco、Pascal context数据集。执行该文件夹下的脚本，可得到相应数据集的分割结果。
对分割结果执行评测文件即可得到最后的miou。
以voc10数据集为例：
1. .json是根据blip，clip预测得到的开放此表标签。如果是弱监督语义分割，则不需要此预测标签。
2. ptp_stable_voc10.py，根据标签预测语义分割结果
3. evaluation_voc10.py 评测语义分割结果
<!-- # 弱监督语义分割
wsss文件夹是弱监督语义分割的代码，使用deeplabv2的代码。设定与clip-es一致。对数据集中的训练集使用我们的方法得到attention（CAM）
再crf后得到训练deeplabv2的伪标签。其余的步骤参考clip-es。 -->


This is code for paper "Diffusion Model is Secretly a Training-free Open Vocabulary Semantic Segmenter"

# Open Vocabulary Semantic Segmentation
The open_vocabulary folder contains code for open vocabulary semantic segmentation. It includes scripts for the voc, coco, and Pascal context datasets. Running scripts in this folder will generate segmentation results for the respective datasets.
To obtain the final mean intersection over union (miou), run the evaluation script on the segmentation results.

Taking the voc10 dataset as an example:
1. .json files contain open vocabulary labels predicted based on blip and clip. If it is weakly supervised semantic segmentation, these predicted labels are not required.
2. ptp_stable_voc10.py is used to predict semantic segmentation results based on the labels.
3. evaluation_voc10.py is used to evaluate the semantic segmentation results.