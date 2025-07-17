#!/bin/bash
#SBATCH -p a30
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --chdir=/remote/gpu03/schiller
#SBATCH --output=/remote/gpu03/schiller/JetCalibration/results/output.txt
#SBATCH --error=/remote/gpu03/schiller/JetCalibration/results/error.txt

my_arg=$1

source activate JetCalibration
cd JetCalibration

export PYTHONPATH=/remote/gpu03/schiller/JetCalibration
python main.py train $my_arg
