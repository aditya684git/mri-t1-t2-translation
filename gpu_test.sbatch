#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --account=<your_account>
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

module load anaconda3/2023.09-0
source activate mri
export PYTHONNOUSERSITE=1

nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x = torch.randn(1000, 1000, device='cuda'); print('matmul ok:', (x @ x).sum().item())"
