#!/bin/bash

source local_env.sh

CUDA_VISIBLE_DEVICES=7 python -m src.runner.inference \
        --config "configs/inference/octopus_8B_s3.yaml"


