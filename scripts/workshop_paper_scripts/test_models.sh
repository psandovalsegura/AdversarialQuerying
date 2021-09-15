#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
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

# ProtoNet miniImageNet 5-shot
python test.py --load /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/mi-protonet-5w5s --way 5 --shot 5 --network ProtoNet --dataset miniImageNet

# ProtoNet CIFAR_FS 5- shot
python test.py --load /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/cifarfs-protonet-5w5s --way 5 --shot 5 --network ProtoNet --dataset CIFAR_FS


# AutoProtoNet miniImageNet 5-shot
python test.py --load /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/mi-autoprotonet-5w5s --way 5 --shot 5 --network AutoProtoNet --dataset miniImageNet

# AutoProtoNet CIFAR_FS 5-shot
python test.py --load /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/cifarfs-autoprotonet-5w5s --way 5 --shot 5 --network AutoProtoNet --dataset CIFAR_FS


# AutoEncoder miniImageNet 5-shot
python test.py --load /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/imagenet_pretrained --way 5 --shot 5 --network AutoProtoNet --dataset miniImageNet

# AutoEncoder CIFAR 5-shot
python test.py --load /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/imagenet_pretrained --way 5 --shot 5 --network AutoProtoNet --dataset CIFAR_FS


# Pretrained then finetuned AutoProtoNet 5-shot
python test.py --load /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/mi-pre-fine-autoprotonet-5w5s --way 5 --shot 5 --network AutoProtoNet --dataset miniImageNet

# Pretrained then finetuned CIFAR-FS 5-shot
python test.py --load /vulcanscratch/psando/checkpoints-meta/workshop_paper_checkpoints/cifarfs-pre-fine-autoprotonet-5w5s --way 5 --shot 5 --network AutoProtoNet --dataset CIFAR_FS
