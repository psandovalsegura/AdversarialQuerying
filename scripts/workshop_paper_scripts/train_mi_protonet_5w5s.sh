#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu

set -x

export WORK_DIR="/scratch0/slurm_${SLURM_JOBID}"
export REPO_DIR="/cfarhomes/psando/Documents/AdversarialQuerying"

# Configure environment
mkdir $WORK_DIR
cd $WORK_DIR
python3 -m venv tmp-env
source tmp-env/bin/activate;pip install --upgrade pip;pip install -r ${REPO_DIR}/requirements-python3.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Train
cd $REPO_DIR
python train.py --train-shot 5 --val-shot 5 --train-way 20 --test-way 5 --train-query 15 --val-query 15 --num-epoch 30 --save-path /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/mi-protonet-5w5s --gpu 0 --network ProtoNet --head ProtoNet --dataset miniImageNet --disable_tqdm