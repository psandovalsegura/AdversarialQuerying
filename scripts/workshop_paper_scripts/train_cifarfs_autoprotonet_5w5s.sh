#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1

set -x

# Configure environment
source /vulcanscratch/psando/envs/official-adv-querying-env/bin/activate

# Train
python train_autoprotonet.py --train-shot 5 --val-shot 5 --train-way 20 --test-way 5 --train-query 15 --val-query 15 --num-epoch 30 --save-path /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/cifarfs-autoprotonet-5w5s --gpu 0 --network AutoProtoNet --head ProtoNet --dataset CIFAR_FS --disable_tqdm