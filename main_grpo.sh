#!/bin/bash

set -x

model_path=/home/fury/models/Qwen2.5-3B #set your model path
train_file_path=data/train/parquet/train_base_enzh_zhen.parquet
test_file_path=data/test/parquet/test_base_enzh_zhen.parquet
comet_model_path=/home/fury/models/XCOMET-XL/checkpoints/model.ckpt #set your metric ckpt
comet_free_model_path=/home/fury/models/COMETKiwi-23-XL/checkpoints/model.ckpt #set your metric ckpt

### Step 1: Process Data
# First run the Python script to prepare the data. 
python3 data/process_data.py \
    --train_files "data/train/json/train_zhen_6565.jsonl" "data/train/json/train_enzh_6565.jsonl" \
    --test_files "data/test/json/wmt23_zhen.jsonl" "data/test/json/wmt24_enzh.jsonl" \
    --tokenizer_path ${model_path} \
    --template_type "reflex" \
    --train_output_file ${train_file_path} \
    --test_output_file ${test_file_path}


### Step 2: MT-R1-Zero Training
export WANDB_API_KEY=7384f1a90be28f8f122dfdeb3f21b3c817c1c331 # set your wandb api key

export VLLM_ATTENTION_BACKEND=XFORMERS
datetime=$(date +"%Y%m%d%H%M%S")
echo $datetime

train_batch_size=8
rollout_num=8
comet_rm=False
comet_free_rm=True 
reward_metric=Merge # []'Model', 'BLEU', 'Merge'] If use reward_metric=BLEU, set comet_rm and comet_free_rm False
exp_name=Reflect-MT-Zero-bs@${train_batch_size}_n@${rollout_num}_comet_rm@${comet_rm}_cometfree_rm@${comet_free_rm}_reward_metric@${reward_metric}_@${datetime}

export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=1,5 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${train_file_path} \
    data.val_files=${test_file_path} \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    comet_model.enable=True \
    comet_model.use_rm=${comet_rm} \
    comet_model.use_valid=True \
    comet_model.ckpt_path=${comet_model_path} \
    comet_free_model.enable=True \
    comet_free_model.use_rm=${comet_free_rm} \
    comet_free_model.use_valid=True \
    comet_free_model.ckpt_path=${comet_free_model_path} \
    algorithm.reward_type='continuous' \
    algorithm.reward_continuous_scale=100 \
    algorithm.reward_metric=${reward_metric} \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.check_think=True \
    trainer.val_before_train=False \
    trainer.logger=['wandb'] \
    trainer.project_name='MT-R1-Zero' \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/home/fury/output/mtzero_runs/${exp_name} \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=1000 \
    trainer.test_freq=200 \
    trainer.total_epochs=1 $@ 2>&1 | tee grpo.log