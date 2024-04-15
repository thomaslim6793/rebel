#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=REBEL_bio_train_on_gpu
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

module load anaconda3/2022.05 cuda/12.1
source ~/miniconda3/etc/profile.d/conda.sh

# Check CUDA availability and version
echo "Checking for CUDA and its version..."
cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \(.*\),.*/\1/')
if [ -z "$cuda_version" ]; then
    echo "CUDA is not available."
    exit 1
else
    echo "CUDA version $cuda_version is available."
fi

# Ensure that the CUDA version is appropriate
if [[ "$cuda_version" != "12.1" ]]; then
    echo "Unexpected CUDA version. Expected 12.1, found $cuda_version."
    exit 1
fi

# Check if the environment exists
conda info --envs | grep cs6120-project
if [ $? -ne 0 ]; then
    echo "Creating a new Conda environment."
    conda create --name cs6120-project python=3.11 -y
else
    echo "Environment cs6120-project already exists."
fi

conda activate cs6120-project
pip install -r requirements.txt

echo "Now running the Python script to start our training process"
export WANDB_MODE=disabled
python src/train.py
