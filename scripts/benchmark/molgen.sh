GPU_IDS=3
export MASTER_PORT=29503

# Load configuration from YAML file
CONFIG_FILE="configs/benchmark/octupus/molgen.yaml"

deepspeed --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} src/runner/sft.py \
    --config ${CONFIG_FILE}