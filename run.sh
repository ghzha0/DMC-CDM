#!/bin/bash
ks=(3 5 10 20)
for k in "${ks[@]}"; do
    echo "set k as: $k"
    sed -i "s/k: .*/k: $k/" ./train_config/model_config.yaml
    CUDA_VISIBLE_DEVICES=2 python run.py
    done
done
