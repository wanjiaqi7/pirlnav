# 是一个 SLURM 作业脚本，用于在具有 GPU 的集群上运行一个名为 "pirlnav" 的作业 
# SLURM(Simple Linux Utility for Resource Management) 是一个开源的集群资源管理器，用于管理大规模的计算集群资源。S

#!/bin/bash
#SBATCH --job-name=pirlnav       ###该作业的作业名（必选）
#SBATCH --gres gpu:4             ###指定GPU资源
#SBATCH --nodes 1                ###该作业需要的节点数（必选）
#SBATCH --cpus-per-task 10       ###每个任务进程所需要的CPU核数，针对多线程任务，默认1（可选）
#SBATCH --ntasks-per-node 4       ###每个节点所运行的进程数（可选）   即每个节点运行4个任务，每个任务分配10个CPU
#SBATCH --signal=USR1@1000        ###作业终止前1000秒（约16分钟）发送一个 USR1 信号给作业，提前通知作业它即将被终止
#SBATCH --partition=short          ###设置分区名（必选）
#SBATCH --constraint=a40           ### 约束条件，指定只在符合 a40 特性（如 GPU 类型为 A40）的节点上运行作业
#SBATCH --output=slurm_logs/ddpil-train-%j.out    ###指定输出文件输出
#SBATCH --error=slurm_logs/ddpil-train-%j.err   ###指定错误文件输出
#SBATCH --requeue          ### 使作业在被中断时（如节点故障或其他原因）重新排队，以便在稍后继续运行

source /srv/flash1/rramrakhya6/miniconda3/etc/profile.d/conda.sh   #  用于加载 Conda 环境的路径
conda deactivate
conda activate pirlnav  # 激活名为 Conda 环境，用于执行训练任务

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

cd /srv/flash1/rramrakhya6/spring_2022/pirlnav   #  切换到 pirlnav 项目的目录

dataset=$1

config="configs/experiments/il_objectnav.yaml"

# DATA_PATH="data/datasets/objectnav/objectnav_hm3d/${dataset}"
# TENSORBOARD_DIR="tb/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"
# CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_10k/"
TENSORBOARD_DIR="tb/objectnav_il/overfitting/ovrl_resnet50/seed_3_wd_zero/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/ovrl_resnet50/seed_3_wd_zero/"
INFLECTION_COEF=3.234951275740812

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x

echo "In ObjectNav IL DDP"
srun python -u -m run \          # srun 命令用于在集群上启动Python脚本。其路径可能是run模块的相对路径。
--exp-config $config \
--run-type train \               # 目的是进行训练，配置了一个训练任务，使用给定的数据集和配置文件
TENSORBOARD_DIR $TENSORBOARD_DIR \
CHECKPOINT_FOLDER $CHECKPOINT_DIR \
NUM_UPDATES 20000 \
NUM_ENVIRONMENTS 16 \
RL.DDPPO.force_distributed True \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF \
