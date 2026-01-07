set -x

export DEBUG_MODE="false"
RUN_NAME="humanIQA_full"
export LOG_PATH="./debug_log_$RUN_NAME.txt"



# set dist args
SINGLE=1

nproc_per_node=8


if [ ! -z "$SINGLE" ] && [ "$SINGLE" != "0" ]; then
  echo "[single node alone] SINGLE=$SINGLE"
  nnodes=1
  node_rank=0
  nproc_per_node=8
  master_addr=127.0.0.1
  master_port=12345
else
  MASTER_NODE_ID=0
  nnodes=${ARNOLD_WORKER_NUM}
  node_rank=${ARNOLD_ID}
  master_addr="METIS_WORKER_${MASTER_NODE_ID}_HOST"
  master_addr=${!master_addr}
  master_port="METIS_WORKER_${MASTER_NODE_ID}_PORT"
  master_port=${!master_port}
  ports=(`echo $master_port | tr ',' ' '`)
  master_port=${ports[0]}
fi

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"


# set up envs
export OMP_NUM_THREADS=6
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=lo
export NCCL_TIMEOUT=180
 
export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

torchrun --nproc_per_node=${nproc_per_node} \
    --nnodes=${nnodes} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
    src/open_r1/humanIQA_full.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir /SAVE_DIR/$RUN_NAME \
    --model_name_or_path  /PRETRAINED_MODEL_PATH \
    --dataset_config data_config/cot_perception.yaml \
    --image_root /KONIQ_DATASET/koniq-10k/512x384\
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing  true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 500 \
    --save_only_model true \
    --score_reward_threshold 0.35 \
    --beta 0.001

