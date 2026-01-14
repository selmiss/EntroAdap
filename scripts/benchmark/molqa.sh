GPU_IDS=6
# export MASTER_PORT=29500

# Load configuration from YAML file
CONFIG_FILE="configs/benchmark/octupus/qa.yaml"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python src/runner/sft.py \
    --config ${CONFIG_FILE}