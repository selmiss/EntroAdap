GPU_IDS=0
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Load configuration from YAML file
CONFIG_FILE="configs/benchmark/octupus/forward.yaml"

python src/runner/sft.py \
    --config ${CONFIG_FILE}