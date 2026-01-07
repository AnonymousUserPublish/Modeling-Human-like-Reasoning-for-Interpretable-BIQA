# Modeling Human-like Reasoning for Intepretable BIQA



## Introduction
HumanIQA aims to guide MLLM to follow a human-like perception-reasoning chain and achieve interpretable and consistent assessment.


##  Dependencies and Installation
```bash
git clone git@github.com:AnonymousUserPublish/Modeling-Human-like-Reasoning-for-Interpretable-BIQA.git
bash setup.sh
```

## Data Preparation 
Download meta files from [Data-DeQA-Score](https://huggingface.co/datasets/zhiyuanyou/Data-DeQA-Score/tree/main) and the source images from the [KONIQ](https://database.mmsp-kn.de/koniq-10k-database.html) dataset.

## Q-Reasoning-Dataset
Source files are in the folder "Q-Reasoning-Dataset",  including the filtered source file (in Japanese) and  Qwen2.5-VL translated and summarized files. (some samples may be automatically removed during filtering). Image sources are from Koniq, SPAQ and Unsplash datasets.




## Pretaining
Please follow the instructions from [Q-Insight](https://github.com/bytedance/Q-Insight).


## Training

```
cd src/open-r1-multimodal/
bash humanIQA_full.sh  # full training
or bash humanIQA_lora.sh  # lora training
```

## Model Weights
Backbone: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and [Q-Insight](https://github.com/bytedance/Q-Insight).
Model weights: waiting for upload.

## File Path
```
PLease change the 1.training config file 2.training dataset folder path 3.default weights path to your own directory.
```

##  To Do List
- [] Release inference code and weights.
- [x] Release training code.
- [x] Release the paper.

## Acknowledgement
This work and repo is built based on [Q-Insight](https://github.com/bytedance/Q-Insight).
We appreciate the releasing codes and data of [Q-Insight](https://github.com/bytedance/Q-Insight),[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [VLM-R1](https://github.com/om-ai-lab/VLM-R1),  and [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score).


## Citation


If you find the code helpful in your research or work, please cite the following papers:
```
@article{2025,
  title={Modeling Human-like Reasoning for Intepretable BIQA},
  year={2025}
}
```







