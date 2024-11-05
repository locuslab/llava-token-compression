#!/bin/bash
#SBATCH --job-name=llava
#SBATCH --output=slurm_output/train_%j.out  # Standard output
#SBATCH --error=slurm_output/train_%j.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=192GB
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00


# #SBATCH --array=0-1%1
# Original training script using Vicuna 13B targets 8 A100 GPUs with 80GB memory. We have... 1. Since we're training the
# 7B parameter model, we can actually fit twice as many samples in memory, so we've increased the per-gpu batch size to
# 64 and the gradient accumulation steps to 4.
# 
# Effective batch size: per-gpu batch size * gradient accumulation steps * number of GPUs
# Original effective batch size: 32 * 1 * 8 = 256
# New effective batch size: 16 * 4 * 4 = 256

# export PATH=/usr/local/cuda-12.2/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda-12.2

source ~/.bashrc
conda activate llava

USER_NAME=kyl2

mkdir -p /scratch/$USER_NAME/LLaVA-Pretrain
rsync -a /data/user_data/sachingo/llava_pretraining_data/LLaVA-Pretrain/ /scratch/$USER_NAME/LLaVA-Pretrain/
PRETRAIN_ROOT=/scratch/$USER_NAME/LLaVA-Pretrain

mkdir -p /scratch/$USER_NAME/LLaVA-Finetune
rsync -a /data/user_data/sachingo/llava_pretraining_data/LLaVA-Finetune/ /scratch/$USER_NAME/LLaVA-Finetune/
FINETUNE_ROOT=/scratch/$USER_NAME/LLaVA-Finetune

# FINETUNE_ROOT=/data/user_data/sachingo/llava_pretraining_data/LLaVA-Finetune/

# declare -a FINAL_TOKEN_COUNTS=(144 64 36 16 4 1)
# declare -a KERNELS=(2 3 4 6 12 24)

# # Get the values for the current task
# FINAL_TOKEN_COUNT=${FINAL_TOKEN_COUNTS[$SLURM_ARRAY_TASK_ID]}
# KERNEL=${KERNELS[$SLURM_ARRAY_TASK_ID]}

KERNEL=4

LLM_VERSION="lmsys/vicuna-7b-v1.5"
LLM_VERSION_SAVE_NAME="vicuna-7b-v1.5"
OUTPUT_ROOT="/data/user_data/kyl2/llava/checkpoints"

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# deepspeed --master_port=$(shuf -i 44000-54000 -n 1) llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path $LLM_VERSION \
#     --version v1 \
#     --data_path ${PRETRAIN_ROOT}/blip_laion_cc_sbu_558k.json \
#     --image_folder ${PRETRAIN_ROOT}/images \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir $OUTPUT_ROOT/${LLM_VERSION_SAVE_NAME}-pretrain-query-embed-local-conv-deep-${KERNEL} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to tensorboard \
#     --mm_vision_token_compression_type query-embed-local-conv-self-attn-deep \
#     --mm_vision_output_text_embedding_size 768 \
#     --mm_vision_output_token_count 576 \
#     --mm_vision_token_compression_kernel_size $KERNEL \
#     --mm_vision_token_compression_stride $KERNEL

# current is 256 samples per update bc 8 grad * 8 bs * 4 gpus 
# previous was 128 per updated bc 4 grad * 8 bs * 4 gpus
# a100_80gb used 16 bs and 4 acc
deepspeed  --master_port=$(shuf -i 44000-54000 -n 1) llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path ${FINETUNE_ROOT}/llava_v1_5_mix665k.json \
    --image_folder ${FINETUNE_ROOT} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $OUTPUT_ROOT/${LLM_VERSION_SAVE_NAME}-pretrain-query-embed-local-conv-deep-${KERNEL}/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_ROOT/${LLM_VERSION_SAVE_NAME}-finetune-query-embed-local-conv-deep-${KERNEL} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --mm_vision_token_compression_type query-embed-local-conv-self-attn-deep \
    --mm_vision_output_text_embedding_size 768 \
    --mm_vision_output_token_count 576 \
    --mm_vision_token_compression_kernel_size $KERNEL \
    --mm_vision_token_compression_stride $KERNEL