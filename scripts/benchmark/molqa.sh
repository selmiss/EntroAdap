GPU_IDS=4
export MASTER_PORT=29504

# Load configuration from YAML file
CONFIG_FILE="configs/benchmark/octupus/qa.yaml"

deepspeed --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} src/runner/sft.py \
    --config ${CONFIG_FILE}