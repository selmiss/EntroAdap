GPU_IDS=7
export MASTER_PORT=29507

# Load configuration from YAML file
CONFIG_FILE="configs/benchmark/llama3/qa.yaml"

deepspeed --include localhost:${GPU_IDS} --master_port ${MASTER_PORT} src/runner/sft.py \
    --config ${CONFIG_FILE}