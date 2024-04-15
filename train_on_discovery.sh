#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --time=08:00:00
#SBATCH --job-name=REBEL_bio_train_on_gpu
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

# Set versions as variables
CUDA_VERSION="11.8"
ANACONDA_VERSION="2022.05"
DESIRED_PYTORCH_VERSION="2.2.2"
DESIRED_CUDA_VERSION="cu118"

module load anaconda3/$ANACONDA_VERSION cuda/$CUDA_VERSION
source ~/miniconda3/etc/profile.d/conda.sh

# Check for NVIDIA Driver version
echo "Checking for NVIDIA Driver version..."
if ! nvidia_driver_version=$(nvidia-smi | grep "Driver Version" | awk '{print $3}'); then
    echo "NVIDIA driver is not available or nvidia-smi not found."
    exit 1
else
    echo "NVIDIA Driver Version: $nvidia_driver_version"
fi

# Check CUDA availability and version in the system
echo "Checking for CUDA and its version..."
if ! cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \(.*\),.*/\1/'); then
    echo "CUDA is not available."
    exit 1
elif [[ "$cuda_version" != "$CUDA_VERSION" ]]; then
    echo "Unexpected CUDA version. Expected $CUDA_VERSION, found $cuda_version."
    exit 1
else
    echo "CUDA version $cuda_version is available."
fi

# Check if the Conda environment exists
echo "Checking Conda environment..."
if ! conda info --envs | grep -q 'cs6120-project'; then
    echo "Creating a new Conda environment."
    conda create --name cs6120-project python=3.11 -y
else
    echo "Environment cs6120-project already exists."
fi

# Activate the Conda environment and install requirements
conda activate cs6120-project

# Check PyTorch and CUDA version
python -c "import torch; print('torch version is: ', torch.__version__); assert torch.__version__ == '$DESIRED_PYTORCH_VERSION+$DESIRED_CUDA_VERSION' and 'cu118' in torch.version.cuda, 'Version mismatch'"

# Only update if assertion fails
if [ $? -ne 0 ]; then
    echo "Updating PyTorch and dependencies to match CUDA 11.8..."
    pip uninstall -y torch
    pip install torch==$DESIRED_PYTORCH_VERSION --index-url https://download.pytorch.org/whl/cu118
fi

pip install -r requirements.txt

echo "Now running the Python script to start our training process"
export WANDB_MODE=disabled
python src/train.py
