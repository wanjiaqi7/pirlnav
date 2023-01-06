#!/bin/bash
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/rl_ft_objectnav.yaml"

TENSORBOARD_DIR="tb/objectnav_il_rl_ft/overfitting/ovrl_resnet50_train_split_hab_v1/seed_1/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il_rl_ft/overfitting/ovrl_resnet50_train_split_hab_v1/seed_1/"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1/"
PRETRAINED_WEIGHTS="data/new_checkpoints/objectnav_il/overfitting/ovrl_resnet50_train_split_hab_v1/seed_1/ckpt.0.pth"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

echo "In ObjectNav IL DDP"
python -u -m run \
--exp-config $config \
--run-type train \
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 1000 \
NUM_PROCESSES 2 \
RL.DDPPO.pretrained_weights $PRETRAINED_WEIGHTS \
RL.DDPPO.distrib_backend "NCCL" \
RL.PPO.num_mini_batch 1 \
RL.Finetune.finetune True \
RL.Finetune.start_actor_update_at 50 \
RL.Finetune.start_actor_warmup_at 50 \
RL.Finetune.start_critic_warmup_at 50 \
RL.Finetune.start_critic_update_at 50 \
RL.PPO.use_linear_lr_decay True \
SENSORS "['RGB_SENSOR']" \
TASK_CONFIG.DATASET.SPLIT "train" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.DATASET.TYPE "ObjectNav-v2" \
TASK_CONFIG.DATASET.MAX_EPISODE_STEPS 500 \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR']" \
