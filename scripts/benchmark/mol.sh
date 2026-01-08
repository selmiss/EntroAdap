GPU_IDS=2
export MASTER_PORT=29502

# Load configuration from YAML file
CONFIG_FILE="configs/benchmark/disease.yaml"

deepspeed --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} src/runner/sft.py \
    --config ${CONFIG_FILE}