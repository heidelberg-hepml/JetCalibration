#!/bin/bash
#PBS -l nodes=1:ppn=10:bigmem1
#PBS -q bigmem1
#PBS -l walltime=10:00:00
#PBS -l mem=200gb,vmem=200gb
#PBD -d /remote/gpu07/huetsch
#PBD -o output.txt
#PBD -e error.txt

cd /remote/gpu07/huetsch
source venv/bin/activate
cd JetCalibration

#pip install uproot
python /remote/gpu07/huetsch/JetCalibration/misc/read_data.py
