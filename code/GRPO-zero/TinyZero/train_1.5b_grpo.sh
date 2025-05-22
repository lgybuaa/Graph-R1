#!/bin/bash
# alias python='/home/weiji/anaconda3/envs/zero/bin/python'
# alias python3='/home/weiji/anaconda3/envs/zero/bin/python3'
# alias pip='/home/weiji/anaconda3/envs/zero/bin/pip'

export N_GPUS=8
ray stop --force && ray start --head --include-dashboard=True
export BASE_MODEL="/home/wuyicong/wyc/graph_reasoning/model/DeepSeek_R1_Distill_Qwen_1.5B"
export DATA_DIR="./graph/data"
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME=grpo-normal-new
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=f1ca3213c83fa01374e225892df98f4bad56b61c
export DEFALUT_LOCAL_DIR="./outputs/grpo-normal-new"

bash ./scripts/train_tiny_zero_a100_grpo.sh