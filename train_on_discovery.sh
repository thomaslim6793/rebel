#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=REBEL_bio_train_on_gpu
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

module load anaconda3/2022.05 cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh
conda create --name cs6120-project python=3.11 -y
conda activate cs6120-project

pip install -r requirements.txt

# Automatically enter '3' to bypass wandb login
export WANDB_MODE=disabled
echo "3" | python src/train.py