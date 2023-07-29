#! /bin/bash

# For CNN-based PCTNet
#model="CNN_pct"
#pretrain_path="./pretrained_models/PCTNet_CNN.pth"

# Uncomment for ViT-based PCTNet 
model="ViT_pct"
pretrain_path="./pretrained_models/PCTNet_ViT.pth"
#pretrain_path="./harmonization_exps/PCTNet_ViT/006/checkpoints/last_checkpoint.pth"
# Evaluation for Full Resolution 
python3 scripts/evaluate_iHarmony4.py ${model} ${pretrain_path} \
    --resize-strategy Fixed256 \
    --res FR \
    --config-path config_test_adobe1024.yml \
    --datasets HAdobe5k \
    --vis-dir \
    /data1/liguanlin/codes/codes_from_github/PCT-Net-Image-Harmonization/results/2048

## Evaluation for Low Resolution
#python3 scripts/evaluate_iHarmony4.py ${model} ${pretrain_path} \
#     --resize-strategy Fixed256 \
#     --res LR \
#     --config-path config.yml 
