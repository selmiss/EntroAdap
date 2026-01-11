# GPU_IDS=0
# export MASTER_PORT=29500

# Load configuration from YAML file
CONFIG_FILE="configs/benchmark/octupus/retro.yaml"

python src/runner/sft.py \
    --config ${CONFIG_FILE}