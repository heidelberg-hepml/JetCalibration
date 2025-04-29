#!/bin/bash
#SBATCH -p a30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=256G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --chdir=/remote/gpu07/huetsch
#SBATCH --output=output.txt
#SBATCH --error=error.txt

my_arg=$1

export CUDA_VISxwIBLE_DEVICES=$(cat $SLURM_JOB_GPUS | sed s/.*-gpu// )
source venv/bin/activate
cd JetCalibration

python main.py plot $my_arg
