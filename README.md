# IdleViT: Efficient Vision Transformer via Token Idle and Token Cut Loss

This repository is the official implementation of [IdleViT: Efficient Vision Transformer via Token Idle and Token Cut Loss]. 

## Requirements

To install requirements:

```setup
conda create -n idle python=3.8
conda activate idle
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install timm
```
## Training

```train
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --output_dir logs/output --arch deit_small --input-size 224 --batch-size 1024 --data-path /path/to/your/dataset --epochs 30 --dist-eval --base_rate 0.7 --lr 1e-5 --min-lr 1e-6 --distill --distillw 10.0 --cutloss_type both
```

## Evaluation

To evaluate my model on ImageNet, run:

For evaluate IdleViT-DeiT-S/0.7
```eval
python infer.py --data-path /path/to/your/dataset --arch deit_small --model-path idlevit_deit_small_ratio0.7.pth --base_rate 0.7
```

For evaluate IdleViT-DeiT-B/0.7
```eval
python infer.py --data-path /path/to/your/dataset --arch deit_base --model-path idlevit_deit_base_ratio0.7.pth --base_rate 0.7
```

For evaluate IdleViT-LV-ViT-S/0.7
```eval
python infer.py --data-path /path/to/your/dataset --arch lvvit_s --model-path idlevit_lvvit_small_ratio0.7.pth --base_rate 0.7
```

For evaluate IdleViT-LV-ViT-M/0.7
```eval
python infer.py --data-path /path/to/your/dataset --arch lvvit_m --model-path idlevit_lvvit_medium_ratio0.7.pth --base_rate 0.7
```

For speed test, taking IdleViT-DeiT-S/0.7 as an example
```eval
python main.py --arch deit_small --input-size 224 --batch-size 128 --data-path ../data/imagenet/ --epochs 30 --dist-eval --distill --base_rate 0.7 --test_speed --only_test_speed
```

## Pre-trained Models

You can download pretrained models here:

- [idlevit_deit_small_ratio0.9.pth](https://drive.google.com/file/d/19RMwxyFTSnLsoQOZ8wdyOLE4DlhG5p1n/view?usp=sharing)
- [idlevit_deit_small_ratio0.8.pth](https://drive.google.com/file/d/1W_AKqDk-PnCZsl7M8FnP5nHfUgj2gFdR/view?usp=sharing)
- [idlevit_deit_small_ratio0.7.pth](https://drive.google.com/file/d/1ZG5a7XoEgLyRiAusQVDH8hIOBWYkgXsR/view?usp=sharing)
- [idlevit_deit_small_ratio0.6.pth](https://drive.google.com/file/d/11fMuuU0Uw-4R-KxXZq2n-R0oW6zbQ6lD/view?usp=sharing)
- [idlevit_deit_small_ratio0.5.pth](https://drive.google.com/file/d/1woPjyWTyxTPEF_ML-G95eELdQic7KRW_/view?usp=sharing)
- [idlevit_deit_base_ratio0.7.pth](https://drive.google.com/file/d/1s0h3LorXW1axd0lkzMAePYut-ZLMI4kB/view?usp=sharing)
- [idlevit_deit_base_ratio0.8.pth](https://drive.google.com/file/d/1CtlBgJJ2MnJwPmWOFWGsr9nlyshLX1sy/view?usp=sharing)
- [idlevit_deit_base_ratio0.9.pth](https://drive.google.com/file/d/1ZaS3GxskWed208hCJUF4jXowdsrbmtwr/view?usp=sharing)
- [idlevit_lvvit_small_ratio0.7.pth](https://drive.google.com/file/d/1ZaS3GxskWed208hCJUF4jXowdsrbmtwr/view?usp=sharing)
- [idlevit_lvvit_medium_ratio0.7.pth](https://drive.google.com/file/d/1Q9ezb5L9SabXlvu7Fh74YjbsYu75AlRz/view?usp=sharing)
